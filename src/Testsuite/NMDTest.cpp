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

#include "NMDTest.h"
#include <blissart/nmf/Deconvolver.h>
#include <blissart/linalg/generators/generators.h>
#include <iostream>
#include <cstdlib>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using namespace blissart::nmf;


namespace Testing {


bool NMDTest::performTest()
{
    srand(1);

    cout << "Creating 10x5 random matrix:" << endl;
    Matrix x(10, 5, nmf::gaussianRandomGenerator);
    cout << x;
    cout << "---" << endl;

    unsigned int t = 3;

    {
        cout << "Performing NMD using KL divergence" << endl;

        nmf::Deconvolver d(x, 10, t);
        d.decompose(nmf::Deconvolver::KLDivergence, 5000, 1e-5);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < t; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Approx = " << endl;
        d.computeApprox();
        Matrix l(d.getApprox());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }
    }
    
    return true;

    {
        cout << "Performing NMD using KL divergence, with normalization" 
             << endl;

        nmf::Deconvolver d(x, 10, t);
        d.decompose(nmf::Deconvolver::KLDivergence, 5000, 1e-5);
        d.normalizeMatrices(nmf::Deconvolver::NormHFrob);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < t; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Approx = " << endl;
        d.computeApprox();
        Matrix l(d.getApprox());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }

        double hfn = d.getH().frobeniusNorm();
        if (!epsilonCheck(hfn, 1.000, 1e-3)) {
            cout << "Error: Expected normalized H, norm is " << hfn << endl;
            return false;
        }
    }

    {
        srand(1);
        cout << "Performing NMD using Itakura-Saito divergence" << endl;

        nmf::Deconvolver d(x, 10, t);
        d.decompose(nmf::Deconvolver::ISDivergence, 5000, 1e-5);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < t; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Approx = " << endl;
        d.computeApprox();
        Matrix l(d.getApprox());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }
    }

    {
        srand(1);
        const double beta = 0.5;
        cout << "Performing NMD using generalized Beta divergence "
             << "(beta = " << beta << ")" << endl;

        nmf::Deconvolver d(x, 10, t);
        d.factorizeNMDBeta(5000, 1e-5, beta, Deconvolver::NoSparsity);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < t; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Approx = " << endl;
        d.computeApprox();
        Matrix l(d.getApprox());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }
    }

    {
        srand(1);
        const double sparsity = 0.25;

        cout << "Performing sparse NMD using KL divergence" << endl;

        nmf::Deconvolver d(x, 10, t);
        //d.decompose(nmf::Deconvolver::KLDivergence, 5000, 1e-5);
        Matrix s(10, x.cols(), generators::unity);
        s.apply(Matrix::mul, sparsity);
        d.setSparsity(s);
        d.factorizeNMDBeta(5000, 1e-5, 1, Deconvolver::NormalizedL1Norm, false);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < t; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Approx = " << endl;
        d.computeApprox();
        Matrix l(d.getApprox());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }
    }

    {
        srand(1);
        const double continuity = 0.1;

        cout << "Performing continuous NMD using KL divergence" << endl;

        nmf::Deconvolver d(x, 10, t);
        //d.decompose(nmf::Deconvolver::KLDivergence, 5000, 1e-5);
        Matrix s(10, x.cols(), generators::unity);
        s.apply(Matrix::mul, continuity);
        d.setContinuity(s);
        d.factorizeNMDBeta(5000, 0.0, 1.0, Deconvolver::NoSparsity, true);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < t; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Approx = " << endl;
        d.computeApprox();
        Matrix l(d.getApprox());
        cout << l << endl;
        
        /*for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }*/
    }

    return true;
}


} // namespace Testing
