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

#include <blissart/audio/MelFilter.h>

#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/Util/Application.h>

#include <cassert>
#include <cmath>


namespace blissart {


using linalg::Matrix;


namespace transforms {


MelFilterTransform::MelFilterTransform(double sampleRate) : 
    _sampleRate(sampleRate), _nBins(0)
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


Matrix* MelFilterTransform::transform(Matrix* spectrogram) const
{
    audio::MelFilter mf(_nBands, _sampleRate, _lowFreq, _highFreq);
    return mf.melSpectrum(*spectrogram);
}


Matrix* MelFilterTransform::inverseTransform(Matrix* melSpectrogram) const
{
    audio::MelFilter mf(_nBands, _sampleRate, _lowFreq, _highFreq);
    Matrix* rv = new Matrix(_nBins, melSpectrogram->cols());
    mf.synth(*melSpectrogram, *rv);
    return rv;
}


const char* MelFilterTransform::name() const
{
    return "Mel filter";
}


} // namespace transforms


} // namespace blissart
