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


#include "GPUMatrixTest.h"
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/GPUMatrix.h>
#include <blissart/linalg/generators/generators.h>

#include <iostream>
#include <cmath>
#include <cstdlib>

//#include <Poco/TemporaryFile.h>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


bool GPUMatrixTest::performTest()
{
    cout << "Creating 2x2 matrix" << endl;
    Matrix a(2, 2, generators::zero);
    a(0,1) = 2;
    a(1,0) = 4;
    cout << a;

    cout << "Creating 2x3 matrix" << endl;
    Matrix b(2, 3, generators::zero);
    b(0,0) = 3;
    b(1,0) = 5;
    b(0,1) = 7;
    b(1,2) = 6;
    cout << b;

    // Matrix * Matrix on CPU
    //const double correctResultMult[] = {10, 0, 12, 12, 28, 0};
    Matrix c(2, 3);
    cout << "---" << endl
         << "Product on CPU:" << endl;
    a.multWithMatrix(b, &c);
    cout << c;


    // Matrix * Matrix on GPU
    GPUMatrix a_d(a);
    Matrix agpu(2,2, generators::zero);
    a_d.getMatrix(&agpu);
    cout << agpu << endl;
    //return true;

    GPUMatrix b_d(b);
    GPUMatrix c_d(c);
    a_d.multWithMatrix(b_d, &c_d);
    Matrix cgpu(2, 3, generators::zero);
    c_d.getMatrix(&cgpu);
    cout << "Product on GPU:" << endl << cgpu;

    srand(1);
    Matrix d(9, 10, generators::random);
    cout << "D = " << endl << d;
    GPUMatrix d_d(d);
    
    // Row sum
    GPUMatrix rsums_d(d.rows(), 1); // "col vector"
    d_d.rowSums(&rsums_d);
    Matrix rsums(d.rows(), 1);
    rsums_d.getMatrix(&rsums);
    cout << "Row sums of D: " << endl;
    cout << rsums << endl;
    for (unsigned int i = 0; i < d.rows(); ++i) {
        //cout << sums(i, 0) << " ";
        if (!epsilonCheck(rsums(i, 0), d.rowSum(i), 1e-6))
            return false;
    }
    
    // Col sum
    GPUMatrix csums_d(1, d.cols()); // "row vector"
    d_d.colSums(&csums_d);
    Matrix csums(1, d.cols());
    csums_d.getMatrix(&csums);
    cout << "Column sums of D: " << endl;
    cout << csums << endl;
    for (unsigned int i = 0; i < d.cols(); ++i) {
        //cout << sums(i, 0) << " ";
        if (!epsilonCheck(csums(0, i), d.colSum(i), 1e-6))
            return false;
    }
    
    // Matrix + Matrix on GPU
    Matrix e(9, 10, generators::random);
    cout << "E = " << endl << e;
    GPUMatrix e_d(e);
    GPUMatrix f_d(9, 10);
    Matrix fgpu(9, 10, generators::zero);
    
    d_d.add(e_d, &f_d);
    f_d.getMatrix(&fgpu);
    cout << "Result D+E: " << endl << fgpu << endl;
    
    d_d.sub(e_d, &f_d);
    f_d.getMatrix(&fgpu);
    cout << "Result D-E: " << endl << fgpu << endl;

    d_d.elementWiseMult(e_d, &f_d);
    f_d.getMatrix(&fgpu);
    cout << "Result D.*E: " << endl << fgpu << endl;

    d_d.elementWiseDiv(e_d, &f_d);
    f_d.getMatrix(&fgpu);
    cout << "Result D./E: " << endl << fgpu << endl;

    d_d.elementWisePow(2.5, &f_d);
    f_d.getMatrix(&fgpu);
    cout << "Result D.^2.5: " << endl << fgpu << endl;
    
    // Scaling
    const double alpha = .5f;
    f_d.scale(alpha, 2, 4);
    f_d.getMatrix(&fgpu);
    cout << "Scale columns 2 to 4 by " << alpha << ":" << endl << fgpu << endl;
    
    // Zero of submatrix
    f_d.zero(2, 3, 7, 8);
    f_d.getMatrix(&fgpu);
    cout << "Set [2,3]->[7,8] to zero:" << endl << fgpu << endl;
    
    // Zero whole matrix
    f_d.zero();
    f_d.getMatrix(&fgpu);
    cout << "Zero matrix:" << endl << fgpu << endl;

    //
    // Large matrix multiplication and verification against CPU gold standard
    //
    
    cout << "Matrix multiplication on CPU ... " << endl;
    int m = 999;
    int k = 199;
    int n = 1;
    Matrix left(m, k, generators::random);
    Matrix right(k, n, generators::unity);
    Matrix resultCPU(m, n);
    left.multWithMatrix(right, &resultCPU);
    
    cout << "Matrix multiplication on GPU ... " << endl;
    GPUMatrix leftGPU(left);
    GPUMatrix rightGPU(right);
    GPUMatrix resultGPU(resultCPU.rows(), resultCPU.cols());
    Matrix    resultGPUtransfer(resultCPU.rows(), resultCPU.cols());
    leftGPU.multWithMatrix(rightGPU, &resultGPU);
    resultGPU.getMatrix(&resultGPUtransfer);
    
    //cout << resultGPUtransfer << endl;
    
    int nwarn = 0;
    for (unsigned int i = 0; i < resultCPU.rows(); ++i) {
        for (unsigned int j = 0; j < resultCPU.cols(); ++j) {
            if (abs(resultCPU(i, j) - resultGPUtransfer(i, j)) > 1e-3) {
                cout << "WARN " << i << " " << j << ": CPU = " << resultCPU(i, j) << "; GPU = " << resultGPUtransfer(i, j) << endl;
                nwarn++;
                if (nwarn > 50)
                    return false;
            }
        }
    }
    
    return true;
}


} // namespace Testing

