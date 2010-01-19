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

#include "SNMFTest.h"
#include <blissart/nmf/Deconvolver.h>
#include <blissart/linalg/generators/generators.h>
#include <iostream>
#include <ctime>
#include <cstdlib>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


bool SNMFTest::performTest()
{
    srand((unsigned int) time(NULL));

    cout << "Creating 10x5 random matrix:" << endl;
    Matrix x(10, 5, nmf::gaussianRandomGenerator);
    cout << x;
    cout << "---" << endl;

    {
        const double sparsity[] = { 0, 0.1, 0.25 };

        for (unsigned int si = 0; si < 3; ++si) {

            cout << "Performing Sparse NMF using KL divergence" << endl;
            cout << "Sparsity parameter set to " << sparsity[si] << endl;

            nmf::Deconvolver d(x, 10, 1);
            Matrix s(10, 5);
            for (unsigned int j = 0; j < s.cols(); ++j) {
                for (unsigned int i = 0; i < s.rows(); ++i) {
                    s(i, j) = sparsity[si];
                }
            }
            d.setS(s);

            d.decompose(nmf::Deconvolver::KLDivergenceSparse, 100, 0.0);
            d.computeApprox();
            cout << "absolute error: " << d.absoluteError() << endl;
            cout << "relative error: " << d.relativeError() << endl;
            cout << endl;

            cout << "W = " << endl;
            cout << d.getW(0) << endl;
            cout << "H = " << endl;
            cout << d.getH() << endl;
            cout << "Approx = " << endl;
            Matrix l(d.getApprox());
            cout << l << endl;
            
            for (unsigned int i = 0; i < x.rows(); i++) {
                for (unsigned int j = 0; j < x.cols(); j++) {
                    if (!epsilonCheck(x(i,j), l(i,j), 5e-2))
                        return false;
                }
            }

        } // for (sparsity param)
    }

    {
        const double sparsity[] = { 0, 0.1, 0.25 };

        for (unsigned int si = 0; si < 3; ++si) {

            cout << "Performing Sparse NMF using Euclidean distance" << endl;
            cout << "Sparsity parameter set to " << sparsity[si] << endl;

            nmf::Deconvolver d(x, 10, 1);
            Matrix s(10, 5);
            for (unsigned int j = 0; j < s.cols(); ++j) {
                for (unsigned int i = 0; i < s.rows(); ++i) {
                    s(i, j) = sparsity[si];
                }
            }
            d.setS(s);

            d.decompose(nmf::Deconvolver::EuclideanDistanceSparse, 1000, 1e-5);
            d.computeApprox();
            cout << "absolute error: " << d.absoluteError() << endl;
            cout << "relative error: " << d.relativeError() << endl;
            cout << endl;

            cout << "W = " << endl;
            cout << d.getW(0) << endl;
            cout << "H = " << endl;
            cout << d.getH() << endl;
            cout << "Approx = " << endl;
            Matrix l(d.getApprox());
            cout << l << endl;
            
            for (unsigned int i = 0; i < x.rows(); i++) {
                for (unsigned int j = 0; j < x.cols(); j++) {
                    if (!epsilonCheck(x(i,j), l(i,j), 5e-2))
                        return false;
                }
            }

        } // for (sparsity param)
    }

    return true;
}


} // namespace Testing
