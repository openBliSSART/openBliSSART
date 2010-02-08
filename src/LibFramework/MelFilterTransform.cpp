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


#include <blissart/transforms/MelFilterTransform.h>
#include <blissart/linalg/Matrix.h>

#include <blissart/feature/mfcc.h>

#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/Util/Application.h>

#include <cassert>
#include <cmath>


namespace blissart {


using linalg::Matrix;


namespace transforms {


MelFilterTransform::MelFilterTransform(double sampleRate) : 
    _sampleRate(sampleRate)
{
    Poco::Util::LayeredConfiguration& cfg 
        = Poco::Util::Application::instance().config();
    _lowFreq = cfg.
        getDouble("blissart.global.mel_filter.low_freq", 0.0);
    _highFreq = cfg.
        getDouble("blissart.global.mel_filter.high_freq", 0.0);
    _nBands = cfg.
        getInt("blissart.global.mel_bands", 26);
}


Matrix* MelFilterTransform::transform(Matrix* spectrogram)
{
    return feature::melSpectrum(*spectrogram, _sampleRate, _nBands);
}


Matrix* MelFilterTransform::inverseTransform(Matrix* melSpectrogram)
{
    // TODO: Implement me!
    return melSpectrogram;
}


const char* MelFilterTransform::name()
{
    return "Mel filter";
}


} // namespace transforms


} // namespace blissart
