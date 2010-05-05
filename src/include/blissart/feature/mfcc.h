//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
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


#ifndef __BLISSART_FEATURE_MFCC_H__
#define __BLISSART_FEATURE_MFCC_H__


#include <cmath>
#include <common.h>


namespace blissart {


// Forward declaration
namespace linalg { class Vector; class Matrix; }


namespace feature {


/**
 * Compute the Mel cepstrum from the given magnitude spectrogram.
 * @param   spectrogram     a reference to a Matrix object containing the
 *                          magnitudes of short-time Fourier spectra
 * @param   sampleRate      the sample rate of the original signal
 * @param   nCoefficients   an integer giving the number of MFCCs to compute
 * @param   nBands          an integer giving the number of filters in the
 *                          filter bank
 * @param   lowFreq         lower cut-off frequency (Hz) of the filter bank
 * @param   highFreq        upper cut-off frequency (Hz) of the filter bank.
                            If set to zero, sampleRate / 2 is assumed.
 * @param   lifter          the value to use for liftering (weighting) of the
 *                          cepstrum
 * @return                  a pointer to a Matrix with Mel spectra in its 
 *                          columns
 */
LibFeature_API linalg::Matrix* 
computeMFCC(const linalg::Matrix& spectrogram, double sampleRate,
            unsigned int nCoefficients, unsigned int nBands = 26,
            double lowFreq = 0.0, double highFreq = 0.0,
            double lifter = 0.0);


/**
 * Convenience function to compute the Mel cepstrum from a single amplitude
 * spectrum.
 * @param   spectrum        a reference to a Vector object containing the
 *                          magnitudes of a short-time Fourier spectrum
 * @param   sampleRate      the sample rate of the original signal
 * @param   nCoefficients   an integer giving the number of MFCCs to compute
 * @param   nBands          an integer giving the number of filters in the
 *                          filter bank
 * @param   lowFreq         lower cut-off frequency (Hz) of the filter bank
 * @param   highFreq        upper cut-off frequency (Hz) of the filter bank.
                            If set to zero, sampleRate / 2 is assumed.
 * @param   lifter          the value to use for liftering (weighting) of the
 *                          cepstrum
 * @return                  a double array containing nCoefficients elements
 */
LibFeature_API double*
computeMFCC(const linalg::Vector& spectrum, double sampleRate,
            unsigned int nCoefficients, unsigned int nBands = 26,
            double lowFreq = 0.0, double highFreq = 0.0,
            double lifter = 0.0);


/**
 * Compute the Mel spectrum from the given magnitude spectrogram.
 * @param   spectrogram     a reference to a Matrix object containing an
 *                          magnitude spectrogram
 * @param   sampleRate      the sample rate of the original signal
 * @param   nBands          an integer giving the number of filters in the
 *                          filter bank
 * @param   lowFreq         lower cut-off frequency (Hz) of the filter bank
 * @param   highFreq        upper cut-off frequency (Hz) of the filter bank.
                            If set to zero, sampleRate / 2 is assumed.
 * @param   scaleFactor     an optional factor to scale the Mel filter output
 * @return                  a pointer to a Matrix object containing Mel spectra
 *                          in its columns
 */
LibFeature_API linalg::Matrix* 
melSpectrum(const linalg::Matrix& spectrogram, double sampleRate,
            unsigned int nBands, double lowFreq = 0.0, double highFreq = 0.0,
            double scaleFactor = 1.0);


/**
 * Compute the cepstrum from the given (mel) spectrogram.
 * @param   melSpectrum     a reference to a Matrix object containing a
 *                          Mel spectrogram
 * @param   nCoefficients   an integer giving the number of MFCCs to compute
 * @param   lifter          the value to use for liftering (weighting) of the
 *                          cepstrum */
LibFeature_API linalg::Matrix* 
computeCepstrogram(const linalg::Matrix& melSpectrum, 
                   unsigned int nCoefficients, double lifter = 0.0);


/**
 * Computes delta regression.
 * @param    coeffMatrix     a reference to a Matrix object containing
 *                           coefficients in its columns
 * @param    theta           extension of the delta window in columns
 */
LibFeature_API linalg::Matrix*
deltaRegression(const linalg::Matrix& coeffMatrix, unsigned int theta);


/**
 * Convert a frequency given in Hz to a frequency in Mel.
 * @param   hertzFrequency  the frequency in Hz
 * @return                  the frequency in Mel
 */
inline double hertzToMel(double hertzFrequency);


/**
 * Convert a frequency given in Mel to a frequency in Hz.
 * @param   melFrequency    the frequency in Mel
 * @return                  the frequency in Hz
 */
inline double melToHertz(double melFrequency);


// Inlines


double hertzToMel(double hertzFrequency)
{
    return 1127.0 * log(1.0 + hertzFrequency / 700.0);
}


double melToHertz(double melFrequency)
{
    return 700.0 * (exp(melFrequency / 1127.0) - 1.0);
}


} }  // namespace blissart::feature


#endif // __BLISSART_FEATURE_MFCC_H__
