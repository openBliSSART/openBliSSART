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


#include <blissart/GnuplotWriter.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/audio/AudioData.h>

#include <fstream>


using blissart::linalg::Matrix;
using namespace std;


namespace blissart {


void GnuplotWriter::writeMatrixGnuplot(const Matrix& m, const string& file,
                                        bool columnWise)
{
    ofstream os(file.c_str());

    // "Column-wise" Gnuplot output.
    if (columnWise) {
        for (unsigned int j = 0; j < m.cols(); ++j) {
            os << j << "\t";
            for (unsigned int i = 0; i < m.rows(); ++i) {
                os << m.at(i, j);
                if (i < m.rows() - 1) os << "\t";
            }
            os << endl;
        }
    }
    // "Row-wise" Gnuplot output.
    else {
        for (unsigned int i = 0; i < m.rows(); ++i) {
            os << i << "\t";
            for (unsigned int j = 0; j < m.cols(); ++j) {
                os << m.at(i, j);
                if (j < m.cols() - 1) os << "\t";
            }
            os << endl;
        }
    }
}


} // namespace blissart

