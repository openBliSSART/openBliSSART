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


#include "ICATest.h"
#include <blissart/ica/FastICA.h>
#include <blissart/linalg/Matrix.h>

#include <Poco/SharedPtr.h>

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>


using namespace blissart;
using namespace blissart::ica;
using namespace blissart::linalg;
using namespace std;
using namespace Poco;


// VC++ complains that fopen & co were deprecated. Since this is Microsoft-Only-stuff
// and also our use of fopen isn't insecure at all we can safely disable the related
// warning:
#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(disable : 4996)
#endif


#define OUTPUT_FILENAME_STAGE1     "1_before_mixing.plt"
#define OUTPUT_FILENAME_STAGE2     "2_after_mixing.plt"
#define OUTPUT_FILENAME_STAGE3     "3_final.plt"


namespace Testing {


static bool saveMatrix(const Matrix& m, const char *fileName)
{
    FILE *fpOut;
    if (!(fpOut = fopen(fileName, "w"))) {
        cerr << "Can't open file " << fileName << "!" << endl;
        return false;
    }

    for (unsigned int j = 0; j < m.cols(); j++) {
        for (unsigned int i = 0; i < m.rows(); i++) {
            fprintf(fpOut, "%f ", m.at(i,j));
        }
        fprintf(fpOut, "\n");
    }

    fclose(fpOut);

    return true;
}


bool ICATest::performTest()
{
    // Create a 3x5000 matrix consisting of samples of two independent
    // random variables. Both random variables are supposed to be
    // equally distributed within [0,1[.
    srand((unsigned)time(NULL));
    Matrix S(3, 5000);
    cout << "Generating random data...";
    for (unsigned int i = 0; i < 5000; i++) {
        S(0,i) = (double)rand() / (double)RAND_MAX;
        S(1,i) = (double)rand() / (double)RAND_MAX;
        S(2,i) = i / 5000.0;
    }
    cout << "done." << endl;

    // Save the raw data samples
    if (!saveMatrix(S, OUTPUT_FILENAME_STAGE1))
        return false;

    // "Mix" the random data (here: rotate and scale)
    const double mix_data[] = { cos(0.7853)/1.3,    -sin(0.7853), 0,
                                sin(0.7853)    , cos(0.7853)/0.5, 0,
                                              0,               0, 1 };
    cout << "Mixing...";
    Matrix X = Matrix(3, 3, mix_data) * S;
    cout << "done." << endl;

    // Save the mixed data samples.
    if (!saveMatrix(X, OUTPUT_FILENAME_STAGE2))
        return false;

    // Construct a FastICA object from the mixed data and perform
    // the corresponding algorithm.
    cout << "Performing FastICA...";
    SharedPtr<FastICA> f = FastICA::compute(&X, 2);
    cout << f->nrOfConvergenceErrors()
         << " convergence errors." << endl;

    // Save the result.
    if (!saveMatrix(X, OUTPUT_FILENAME_STAGE3))
        return false;

    if (f->nrOfConvergenceErrors() > 0)
        return false;

    return true;
}


} // namespace Testing
