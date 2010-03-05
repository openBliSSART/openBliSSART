//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
//                          Felix Weninger <felix@weninger.de>
//                          Bjoern Schuller <schuller@tum.de>
//
// Institute for Human-Machine Communication
// Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
//
// openBliSSART is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 2 of the License, or (at your option) any later
// version.
//
// openBliSSART is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
//


#include <blissart/feature/mfcc.h>
#include <blissart/linalg/Vector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>

#include <cmath>
#include <cassert>


using namespace blissart::linalg;


namespace blissart {

namespace feature {


Matrix* computeMFCC(const Matrix& spectrogram,
                    double sampleRate, unsigned int nCoefficients,
                    unsigned int nBands, double lowFreq, double highFreq, 
                    double lifter)
{
    Matrix* s = melSpectrum(spectrogram, sampleRate, nBands, 
        lowFreq, highFreq);
    Matrix* rv = computeCepstrogram(*s, nCoefficients, lifter);
    delete s;
    return rv;
}


double* computeMFCC(const Vector& spectrum,
                    double sampleRate, unsigned int nCoefficients,
                    unsigned int nBands, double lowFreq, double highFreq,
                    double lifter)
{
    Matrix spectrogram(spectrum.dim(), 1);
    for (unsigned int i = 0; i < spectrum.dim(); ++i) {
        spectrogram.at(i, 0) = spectrum.at(i);
    }
    Matrix* s = melSpectrum(spectrogram, sampleRate, nBands, 
        lowFreq, highFreq);
    Matrix* temp = computeCepstrogram(*s, nCoefficients, lifter);
    delete s;
    double* rv = new double[nCoefficients];
    for (unsigned int i = 0; i < temp->rows(); ++i) {
        rv[i] = temp->at(i, 0);
    }
    delete temp;
    return rv;
}


inline double round(double x) 
{
    return (x > 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}


Matrix* melSpectrum(const Matrix& spectrogram,
                    double sampleRate, unsigned int nBands,
                    double lowFreq, double highFreq,
                    double scaleFactor)
{
    assert(spectrogram.rows() >= nBands);
    assert(sampleRate > 0.0);
    assert(lowFreq >= 0.0 && highFreq >= 0.0);

    if (highFreq == 0.0) highFreq = sampleRate / 2.0;

    // Precompute basic parameters.
    unsigned int nSamples       = (spectrogram.rows() - 1) * 2;
    const double baseFreq       = sampleRate / nSamples;
    const double lowestMelFreq  = hertzToMel(lowFreq);
    const double highestMelFreq = hertzToMel(highFreq);
    unsigned int lowestIndex    = (unsigned int) round(lowFreq / baseFreq);
    unsigned int highestIndex   = (unsigned int) round(highFreq / baseFreq);
    
    // Always ignore zeroth FFT coefficient (DC component).
    if (lowestIndex < 1) lowestIndex = 1;
    
    assert(highestIndex < spectrogram.rows() && 
           lowestIndex < spectrogram.rows());

    // Compute Mel center frequencies
    double* centerFrequencies = new double[nBands + 2];
    const double halfBw = highestMelFreq / ((double)nBands + 1.0);
    int m = 0;
    for (; m <= (int)nBands + 1; ++m) {
        // Distance between center frequencies is half the Mel bandwidth of
        // a filter.
        centerFrequencies[m] = lowestMelFreq + (double)m * halfBw;
    }

    // Compute the index of the filter that is to be applied to every component
    // of the spectrum. Note that the falling slope of filter M is equal to 
    // the rising slope of filter M+1, which is why we calculate falling slopes
    // only - thus the filter index can have a value of -1. A value of -2 
    // (occurring at the boundaries of the filter bank) indicates that the 
    // filter output is zero.
    int* filterIndex = new int[spectrogram.rows()];
    m = 0;
    for (unsigned int i = 0; i < spectrogram.rows(); ++i) {
        if (i < lowestIndex || i > highestIndex) {
            // XXX: -3?
            filterIndex[i] = -3;
        }
        else {
            double binFreq = hertzToMel((double)i * baseFreq);
            while (m <= (int)nBands + 1 && centerFrequencies[m] < binFreq) {
                ++m;
            }
            filterIndex[i] = m - 2;
        }
    }

    // Compute filter coefficients (for falling slopes).
    double* filterCoeffs = new double[spectrogram.rows()];
    m = 0;
    for (unsigned int i = lowestIndex; i < highestIndex; ++i) {
        double binFreq = hertzToMel((double)i * baseFreq);
        while (m <= (int)nBands && binFreq > centerFrequencies[m + 1]) {
            ++m;
        }
        filterCoeffs[i] = (centerFrequencies[m + 1] - binFreq) /
                          (centerFrequencies[m + 1] - centerFrequencies[m]);
    }

    // Free memory.
    delete[] centerFrequencies;

    // Process matrix column by column.
    Matrix* rv = new Matrix(nBands, spectrogram.cols(), generators::zero);
    for (unsigned int j = 0; j < spectrogram.cols(); ++j) {
        for (unsigned int i = lowestIndex; i < highestIndex; ++i) {
            m = filterIndex[i];
            if (m > -2) {
                double out = spectrogram.at(i, j) * filterCoeffs[i];
                if (m > -1) {
                    rv->at(m, j) += out;
                }
                if (m < (int) nBands - 1) {
                    rv->at(m + 1, j) += spectrogram.at(i, j) - out;
                }
            }
        }
        if (scaleFactor != 1.0) {
            for (m = 0; m < (int)nBands; ++m) {
                rv->at(m, j) *= scaleFactor;
            }
        }
    }

    // Free memory.
    delete[] filterCoeffs;
    delete[] filterIndex;

    return rv;
}


Matrix* computeCepstrogram(const Matrix& melSpectrum, 
                           unsigned int nCoefficients, double lifter)
{
    static const double pi = 4.0 * atan(1.0);
    Matrix* rv = new Matrix(nCoefficients, melSpectrum.cols());
    const double normalization = sqrt(2.0 / (double)melSpectrum.rows());
    for (unsigned int j = 0; j < melSpectrum.cols(); ++j) {
        // Compute DCT. Since we do not need all coefficients, simply use
        // own implementation instead of fftw.
        for (unsigned int i = 0; i < nCoefficients; ++i) {
            double dctCoeff = 0.0;
            for (unsigned int m = 0; m < melSpectrum.rows(); ++m) {
                double x = pi * (double)i / (double)melSpectrum.rows() * 
                    ((double) m + 0.5);
                if (melSpectrum.at(m, j) < 1e-6) {
                    dctCoeff += cos(x) * log(1e-6);
                }
                else {
                    dctCoeff += cos(x) * log(melSpectrum(m, j));
                }
            }
            dctCoeff *= normalization;
            if (lifter > 0.0) {
                dctCoeff *= (1.0 + lifter / 2.0 * sin(pi * (double)i / lifter));
            }
            rv->setAt(i, j, dctCoeff);
        }
    }

    return rv;
}


Matrix* deltaRegression(const Matrix& coeffMatrix, unsigned int theta)
{
    Matrix* rv = new Matrix(coeffMatrix.rows(), coeffMatrix.cols(),
        generators::zero);

    double normalization = 0.0;
    for (unsigned int t = 1; t <= theta; ++t) {
        normalization += (t * t);
    }
    normalization *= 2.0;

    for (unsigned int j = 0; j < coeffMatrix.cols(); ++j) {
        for (unsigned int i = 0; i < coeffMatrix.rows(); ++i) {
            for (unsigned int t = 1; t <= theta; ++t) {
                double left  = j < t ? 
                    coeffMatrix.at(i, 0) : 
                    coeffMatrix.at(i, j - t);
                double right = j + t >= coeffMatrix.cols() ? 
                    coeffMatrix.at(i, coeffMatrix.cols() - 1) : 
                    coeffMatrix.at(i, j + t);
                rv->at(i, j) += t * (right - left);
            }
            rv->at(i, j) /= normalization;
        }
    }
    
    return rv;
}


} // namespace feature

} // namespace blissart

