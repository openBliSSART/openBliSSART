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


#include "Testable.h"
#include <blissart/linalg/Matrix.h>


using blissart::linalg::Matrix;


namespace Testing {


bool Testable::epsilonCheck(const Matrix& a, const Matrix& b, 
                            const double prec)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        return false;

    for (unsigned int j = 0; j < a.cols(); ++j) {
        for (unsigned int i = 0; i < a.rows(); ++i) {
            if (!epsilonCheck(a(i, j), b(i, j), prec)) {
                return false;
            }
        }
    }

    return true;
}


} // namespace Testing
