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


#include <blissart/transforms/PowerTransform.h>
#include <blissart/linalg/Matrix.h>

#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/Util/Application.h>

#include <cassert>
#include <cmath>


namespace blissart {


using linalg::Matrix;


namespace transforms {


PowerTransform::PowerTransform()
{
    Poco::Util::LayeredConfiguration& cfg 
        = Poco::Util::Application::instance().config();
    _gamma = 
        cfg.getDouble("blissart.fft.transformations.powerSpectrum.gamma", 2.0);
}


Matrix* PowerTransform::transform(Matrix* spectrogram) const
{
    powMatrix(spectrogram, _gamma);
    return spectrogram;
}


Matrix* PowerTransform::inverseTransform(Matrix* spectrogram) const
{
    powMatrix(spectrogram, 1.0 / _gamma);
    return spectrogram;
}


void PowerTransform::powMatrix(Matrix* m, double exp) const
{
    for (unsigned int j = 0; j < m->cols(); ++j) {
        for (unsigned int i = 0; i < m->rows(); ++i) {
            m->at(i, j) = std::pow(m->at(i, j), exp);
        }
    }
}


const char* PowerTransform::name() const
{
    return "Power spectrum";
}


} // namespace transforms


} // namespace blissart
