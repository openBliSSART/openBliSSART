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

#include "SNMFTest.h"
#include <blissart/nmf/Deconvolver.h>
#include <blissart/linalg/generators/generators.h>
#include <iostream>
#include <ctime>
#include <cstdlib>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using nmf::Deconvolver;


namespace Testing {


bool SNMFTest::performTest()
{
    srand((unsigned int) time(NULL));

    cout << "Creating 10x5 random matrix:" << endl;
    Matrix x(10, 5, nmf::gaussianRandomGenerator);
    cout << x;
    cout << "---" << endl;

    // This sanity check tests whether a sparse NMF with the sparseness
    // weight set to zero equals normal NMF. This test is necessary since
    // implementation might select different algorithms for both tasks.
    // In this test, of course, the same initialization has to be chosen
    // for both variants.

    cout << "Performing Sparse NMF sanity check" << endl << endl;

    {
        cout << "Euclidean distance" << endl;
        nmf::Deconvolver d(x, 10, 1);
        Matrix w(10, 10, nmf::gaussianRandomGenerator);
        Matrix h(10, 5, nmf::gaussianRandomGenerator);
        d.setW(0, w);
        d.setH(h);
        d.decompose(Deconvolver::EuclideanDistanceSparse, 1, 0.0);
        d.computeApprox();
        Matrix wh1(d.getApprox());
        d.setW(0, w);
        d.setH(h);
        d.decompose(Deconvolver::EuclideanDistance, 1, 0.0);
        d.computeApprox();
        Matrix wh2(d.getApprox());
        cout << "1 iteration of sparse NMF (lambda = 0)" << endl;
        cout << wh1;
        cout << "1 iteration of NMF" << endl;
        cout << wh2;
        if (!epsilonCheck(wh1, wh2, 1e-2)) {
            return false;
        }
    }

    {
        cout << "KL divergence" << endl;
        nmf::Deconvolver d(x, 10, 1);
        Matrix w(10, 10, nmf::gaussianRandomGenerator);
        Matrix h(10, 5, nmf::gaussianRandomGenerator);
        d.setW(0, w);
        d.setH(h);
        d.decompose(Deconvolver::KLDivergenceSparse, 1, 0.0);
        d.computeApprox();
        Matrix wh1(d.getApprox());
        d.setW(0, w);
        d.setH(h);
        d.decompose(Deconvolver::KLDivergence, 1, 0.0);
        d.computeApprox();
        Matrix wh2(d.getApprox());
        cout << "1 iteration of sparse NMF (lambda = 0)" << endl;
        cout << wh1;
        cout << "1 iteration of standard NMF" << endl;
        cout << wh2;
        if (!epsilonCheck(wh1, wh2, 1e-2)) {
            return false;
        }
    }

    // Convergence tests for different sparsity parameters and both cost 
    // functions.

    {
        const double sparsity[] = { 0, 0.1, 0.25 };

        for (unsigned int si = 0; si < 3; ++si) {

            cout << endl << "---" << endl;
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

    {
        const double sparsity[] = { 0, 0.1, 0.25 };

        for (unsigned int si = 0; si < 3; ++si) {

            cout << "Performing Sparse NMF using Euclidean distance, normalized basis" << endl;
            cout << "Sparsity parameter set to " << sparsity[si] << endl;

            nmf::Deconvolver d(x, 10, 1);
            Matrix s(10, 5);
            for (unsigned int j = 0; j < s.cols(); ++j) {
                for (unsigned int i = 0; i < s.rows(); ++i) {
                    s(i, j) = sparsity[si];
                }
            }
            d.setS(s);

            d.decompose(nmf::Deconvolver::EuclideanDistanceSparseNormalized, 1000, 1e-5);
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
                    // this variant of SNMF seems to be less precise
                    // concerning reconstruction error!
                    if (!epsilonCheck(x(i,j), l(i,j), 2e-1))
                        return false;
                }
            }

        } // for (sparsity param)
    }

    return true;
}


} // namespace Testing
