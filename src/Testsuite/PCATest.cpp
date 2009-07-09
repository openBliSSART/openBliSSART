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


#include "PCATest.h"
#include <blissart/ica/PCA.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>

#include <iostream>
#include <memory>


using namespace blissart;
using namespace blissart::ica;
using namespace blissart::linalg;
using namespace std;


namespace Testing {


bool PCATest::performTest()
{
    const double small_data[] = {   2, 3.5, 2.5,    0, -1.5, -0.5,
                                  0.5,   0,   2, -0.5,    0,   -2,
                                    3,   3,   3,    3,    3,    3};
    Matrix M(3, 6, small_data);

    cout << "---" << endl
         << "Matrix M:" << endl << M;

    auto_ptr<PCA> p(PCA::compute(&M, 2, false));

    // Mean vector
    cout << "Mean vector: " << p->expectedValue() << endl;

    // Covariance matrix
    const double correct_cov[] = { 3.8, 1.4, 0,
                                   1.4, 1.7, 0,
                                     0,   0, 0 };
    cout << "Covariance matrix:" << endl
         << p->covarianceMatrix();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (!epsilonCheck(p->covarianceMatrix().at(i,j), correct_cov[i*3+j]))
                return false;
        }
    }
           
    // Eigenpairs
    cout << "Eigenpairs:" << endl;
    for (Matrix::EigenPairs::const_iterator it = p->eigenPairs().begin();
        it != p->eigenPairs().end(); ++it)
    {
        cout << "\t" << it->first << " -> " << it->second << endl;
    }
    if (p->eigenPairs().size() != 2 ||
        !epsilonCheck(p->eigenPairs().at(0).first, 4.5) ||
        !epsilonCheck(p->eigenPairs().at(1).first, 1))
    {
        return false;
    }

    cout << "After PCA:" << endl << M;

    // IMPORTANT: For eigenvectors as "basis" it doesn't matter whether they
    // point in direction a or -a. Therefore it happens (depending on random
    // number generation during PCA) that this test result shows false although
    // it is correct. If it does show false one has to check manually if the
    // result makes sense (sorry).
    const double correct_result1[] = { 1.12,  2.24, 2.24, -1.12, -2.24, -2.24,
                                          0, -1.12, 1.12,     0,  1.12, -1.12 };
    const double correct_result2[] = { 1.12, 2.24,  2.24, -1.12, -2.24, -2.24,
                                          0, 1.12, -1.12,     0, -1.12,  1.12 };
    bool correct1 = true;
    bool correct2 = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            if (!epsilonCheck(M(i,j), correct_result1[i*6 + j], 1e-2))
                correct1 = false;
            if (!epsilonCheck(M(i,j), correct_result2[i*6 + j], 1e-2))
                correct2 = false;
        }
    }
    if (!correct1 && !correct2)
        return false;

    return true;
}


} // namespace Testing
