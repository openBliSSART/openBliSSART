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

#include "NMDTest.h"
#include <blissart/nmf/Deconvolver.h>
#include <blissart/linalg/generators/generators.h>
#include <iostream>
#include <ctime>
#include <cstdlib>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


bool NMDTest::performTest()
{
    srand((unsigned int) time(NULL));

    cout << "Creating 10x5 random matrix:" << endl;
    Matrix x(10, 5, nmf::gaussianRandomGenerator);
    /*double norm = sqrt(x.dotColCol(x, 0, x, 0));
    for (unsigned int i = 0; i < x.rows(); ++i) {
        x(i, 0) /= norm;
    }
    cout << x;*/
    cout << "---" << endl;

    const unsigned int t = 3;

    {
        cout << "Performing NMD using KL divergence" << endl;

        nmf::Deconvolver d(x, 10, t);
        d.factorizeKL(5000, 1e-5);
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
        cout << "Lambda = " << endl;
        d.computeLambda();
        Matrix l(d.getLambda());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 1e-2))
                    return false;
            }
        }
    }

    {
        cout << "Performing NMD using Euclidean distance" << endl;

        nmf::Deconvolver d(x, 10, t);
        d.factorizeED(5000, 1e-5);
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
        cout << "Lambda = " << endl;
        d.computeLambda();
        Matrix l(d.getLambda());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 5e-2))
                    return false;
            }
        }
    }

    {
        cout << "Performing NMF using Euclidean distance, with normalization" 
             << endl;

        nmf::Deconvolver d(x, 10, 1);
        d.setNormalizeW(true);
        d.factorizeED(5000, 1e-5);
        cout << "# steps: " << d.numSteps() << endl;
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;
        for (unsigned int i = 0; i < 1; ++i) {
            cout << "W[" << i << "] = " << endl;
            cout << d.getW(i) << endl;
        }
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "Lambda = " << endl;
        d.computeLambda();
        Matrix l(d.getLambda());
        cout << l << endl;
        
        for (unsigned int i = 0; i < x.rows(); i++) {
            for (unsigned int j = 0; j < x.cols(); j++) {
                if (!epsilonCheck(x(i,j), l(i,j), 5e-2))
                    return false;
            }
        }
    }

    return true;
}


} // namespace Testing
