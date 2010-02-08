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


#include <blissart/transforms/SlidingWindowTransform.h>
#include <blissart/linalg/Matrix.h>

#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/Util/Application.h>

#include <cassert>
#include <cmath>


namespace blissart {


using linalg::Matrix;


namespace transforms {


SlidingWindowTransform::SlidingWindowTransform()
{
    Poco::Util::LayeredConfiguration& cfg 
        = Poco::Util::Application::instance().config();
    _frameSize = 
        cfg.getInt("blissart.fft.transformations.slidingWindow.frameSize", 10);
    _frameRate =
        cfg.getInt("blissart.fft.transformations.slidingWindow.frameRate", 1);
    assert(_frameSize > 0);
}


Matrix* SlidingWindowTransform::transform(Matrix* spectrogram) const
{
    unsigned int nCols = (unsigned int) 
        std::ceil(((double)spectrogram->cols() - _frameSize) / _frameRate) + 1;
    Matrix* output = new Matrix(spectrogram->rows() * _frameSize, nCols);
    unsigned int frameStart = 0;
    for (unsigned int outputCol = 0; outputCol < nCols; 
        ++outputCol, frameStart += _frameRate) 
    {
        unsigned int outputRow = 0;
        for (unsigned int pos = 0; pos < _frameSize; ++pos)
        {
            for (unsigned int inputRow = 0; inputRow < spectrogram->rows(); 
                ++inputRow, ++outputRow)
            {
                output->at(outputRow, outputCol) 
                    = spectrogram->at(inputRow, frameStart + pos);
            }
        }
    }
    return output;
}


Matrix* SlidingWindowTransform::inverseTransform(Matrix* spectrogram) const
{
    // TODO: Implement me!
    return spectrogram;
}


const char* SlidingWindowTransform::name() const
{
    return "Sliding window spectrum";
}


} // namespace transforms


} // namespace blissart
