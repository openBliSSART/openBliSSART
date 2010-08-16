
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


#include <blissart/transforms/SpectralSubtractionTransform.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>

#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/Util/Application.h>
#include <Poco/NumberFormatter.h>

#include <cassert>
#include <cmath>
#include <vector>


namespace blissart {


using linalg::Matrix;
using linalg::RowVector;


namespace transforms {


SpectralSubtractionTransform::SpectralSubtractionTransform()
{
}


Matrix* SpectralSubtractionTransform::transform(Matrix* spectrogram) const
{
    // TODO: add proper VAD etc.!

    // get noise "model"
    RowVector noise(spectrogram->rows());
    unsigned int noisec = 10;
    if (spectrogram->cols() < noisec)
        noisec = spectrogram->cols();
    
    for (unsigned int i = 0; i < noise.dim(); ++i) {
        noise(i) = spectrogram->rowSum(i, 0, noisec - 1) / noisec;
    }

    // subtraction
    for (unsigned int j = 0; j < spectrogram->cols(); ++j) {
        for (unsigned int i = 0; i < spectrogram->rows(); ++i) {
            spectrogram->at(i, j) -= noise(i);
            // thresholding
            if (spectrogram->at(i, j) < 0)
                spectrogram->at(i, j) = 0;
        }
    }

    return spectrogram;
}


Matrix* SpectralSubtractionTransform::inverseTransform(Matrix* spectrogram) const
{
    // identity
    return spectrogram;
}


const char* SpectralSubtractionTransform::name() const
{
    return "Spectral subtraction";
}


} // namespace transforms


} // namespace blissart

