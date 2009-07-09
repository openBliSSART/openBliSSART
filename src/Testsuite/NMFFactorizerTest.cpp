//
// $Id: NMFFactorizerTest.cpp 855 2009-06-09 16:15:50Z alex $
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

#include "NMFFactorizerTest.h"
#include <blissart/nmf/Factorizer.h>
#include <blissart/linalg/generators/generators.h>
#include <iostream>
#include <cstdlib>
#include <ctime>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


double rgen1(unsigned int, unsigned int)
{
    return 1.0 + (double) rand() / (double) RAND_MAX;
}


double rgen2(unsigned int, unsigned int)
{
    return (1.0 + (double) rand() / (double) RAND_MAX) * 1e-2;
}


bool NMFFactorizerTest::performTest()
{
    cout << "Creating 10x5 random matrix:" << endl;
    Matrix x(10, 5, generators::random);
    cout << x;
    cout << "---" << endl;

    nmf::Factorizer f(x, 10);

    cout << "Performing NMF using distance algorithm" << endl;
    f.factorizeDistance(5000, 1e-3);
    cout << "# steps: " << f.numSteps() << endl;
    cout << "absolute error: " << f.absoluteError() << endl;
    cout << "relative error: " << f.relativeError() << endl;
    cout << endl;
    cout << "W = " << endl;
    Matrix w = f.getFirst();
    cout << w << endl;
    cout << "H = " << endl;
    Matrix h = f.getSecond();
    cout << h << endl;
    cout << "WH = " << endl;
    Matrix x2 = w * h;
    cout << x2 << endl;
    
    for (unsigned int i = 0; i < x.rows(); i++) {
        for (unsigned int j = 0; j < x.cols(); j++) {
            if (!epsilonCheck(x(i,j), x2(i,j), 1e-2))
                return false;
        }
    }

    cout << "Performing NMF using divergence algorithm" << endl;
    // We have to reset the factors, otherwise we already start with a quite
    // good solution...
    f.randomizeFirst();
    f.randomizeSecond();
    f.factorizeDivergence(5000, 1e-3);
    cout << "# steps: " << f.numSteps() << endl;
    cout << "absolute error: " << f.absoluteError() << endl;
    cout << "relative error: " << f.relativeError() << endl;
    cout << endl;
    cout << "W = " << endl;
    w = f.getFirst();
    cout << w << endl;
    cout << "H = " << endl;
    h = f.getSecond();
    cout << h << endl;
    cout << "WH = " << endl;
    w.multWithMatrix(h, &x2);
    cout << x2 << endl;

    for (unsigned int i = 0; i < x.rows(); i++) {
        for (unsigned int j = 0; j < x.cols(); j++) {
            if (!epsilonCheck(x(i,j), x2(i,j), 1e-2))
                return false;
        }
    }

    cout << endl;
    cout << "-------------------------" << endl;
    cout << "Demonstration of impact of different initialisation" << endl;
    cout << endl;
    cout << "Initialization with ]1, 2] values" << endl;
    f.setFirst(Matrix(10, 10, &rgen1));
    f.setSecond(Matrix(10, 5, &rgen1));
    f.factorizeDivergence(5000, 1e-3);
    cout << "# steps: " << f.numSteps() << endl;
    cout << "W = " << endl;
    cout << f.getFirst() << endl;
    cout << "H = " << endl;
    cout << f.getSecond() << endl;

    cout << endl;
    cout << "Initialization with ]0.01, 0.02] values" << endl;
    f.setFirst(Matrix(10, 10, &rgen2));
    f.setSecond(Matrix(10, 5, &rgen2));
    f.factorizeDivergence(5000, 1e-3);
    cout << "# steps: " << f.numSteps() << endl;
    cout << "W = " << endl;
    cout << f.getFirst() << endl;
    cout << "H = " << endl;
    cout << f.getSecond() << endl;

    return true;
}


} // namespace Testing
