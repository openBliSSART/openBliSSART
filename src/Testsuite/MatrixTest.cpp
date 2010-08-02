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


#include "MatrixTest.h"
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/generators/generators.h>

#include <iostream>
#include <cmath>

#include <Poco/TemporaryFile.h>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


//#ifdef ISEP_ROW_MAJOR
//# error The matrix tests depend on !ISEP_ROW_MAJOR. See Matrix.h.
//#endif


namespace Testing {


static double matrixGenerator(unsigned int i, unsigned int j)
{
    return i * j;
}


bool MatrixTest::performTest()
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

    double fn = b.frobeniusNorm();
    cout << "Frobenius norm: " << fn << endl;
    if (!epsilonCheck(fn, 10.9, 1e-2))
        return false;

    // Matrix * Matrix
    const double correctResultMult[] = {10, 0, 12, 12, 28, 0};
    Matrix c(2, 3);
    cout << "---" << endl
         << "Product:" << endl;
    a.multWithMatrix(b, &c);
    cout << c;
    if (!(Matrix(2, 3, correctResultMult) == c))
        return false;

    // Transpose
    const double correctResultTranspose[] = {10, 12, 0, 28, 12, 0};
    Matrix cT(3, 2);
    cout << "---" << endl
         << "Transpose of product:" << endl;
    c.transpose(&cT);
    cout << cT;
    if (!(Matrix(3, 2, correctResultTranspose) == cT))
        return false;

    // Random matrix
    cout << "---" << endl
         << "Random matrix:" << endl;
    Matrix d(4, 4, generators::random);
    cout << d;

    // Shifts
    {
        const double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        const double datar[] = {0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11};
        const double datal[] = {2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 0};
        Matrix e(3, 4, data);
        Matrix er(e);
        er.shiftColumnsRight();
        cout << "---" << endl
             << "Matrix E: " << endl << e << endl
             << "Shifted by 1 column to the right:" << endl << er << endl;
        if (Matrix(3, 4, datar) != er) {
            return false;
        }
        Matrix el(e);
        el.shiftColumnsLeft();
        cout << "Shifted by 1 column to the left:" << endl << el << endl;
        if (Matrix(3, 4, datal) != el) {
            return false;
        }
    }

    // Use of generator function
    {
        cout << "---" << endl
             << "Use of generator function...";
        Matrix g(10, 10, matrixGenerator);
        for (unsigned int i = 0; i < 10; i++) {
            for (unsigned int j = 0; j < 10; j++) {
                if (g(i,j) != i * j)
                    return false;
            }
        }
        cout << "ok." << endl;
    }

    // Gaussian Elimination
    {
        cout << "---" << endl
             << "Matrix A before Gaussian Elimination:" << endl;
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        cout << A << endl
             << "Matrix A after Gaussian Elimination:" << endl
             << A
             << "Swapped Rows: " << A.gaussElimination() << endl;
    }

    // LSE
    {
        cout << "---" << endl
             << "Matrix A:" << endl;
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        cout << A << endl;
        const double v_data[] = {42,72,-86};
        const ColVector v(3, v_data);
        ColVector target(3);
        cout << "v := " << v << "^T. Solve Ax = v:" << endl;
        Matrix::linearSolve(A, v, &target);
        cout << "x := (" << target(0) << "," << target(1) << ","
             << target(2) << ")" << endl;
        if (!epsilonCheck(target(0), 3)
            || !epsilonCheck(target(1), 2)
            || !epsilonCheck(target(2), 5))
            return false;
    }

    // Determinant 3x3
    {
        cout << "---" << endl
             << "Matrix A:" << endl;
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        cout << A << endl;
        const double det = A.determinant();
        cout << "Determinant of A: " << det << endl;
        if (!epsilonCheck(det, 1248))
            return false;
    }

    // Determinant 4x4
    {
        cout << "---" << endl
             << "Matrix A:" << endl;
        double A_data[] = {  1,  4,   7, 2,
                             5,  8,   3, 6,
                           -23, -2,   0, 8,
                             7,  1, -14, 3 };
        Matrix A(4,4,A_data);
        cout << A << endl;
        const double det = A.determinant();
        cout << "Determinant of A: " << det << endl;
        if (!epsilonCheck(det, 696))
            return false;
    }

    // Trace
    {
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        const double trace = A.trace();
        cout << "---" << endl
             << "Matrix A:" << endl << A
             << "Trace(A) = " << trace << endl;
        if (!epsilonCheck(trace, 5))
            return false;
    }

    // Multiplication with a column vector
    {
        const double tm_data[] = {10, 0, 12, 12, 28, 0};
        Matrix tm(2, 3, tm_data);
        cout << "---" << endl
             << "Matrix tm:" << endl << tm;
        const double vec_data[] = {1, -7, 5};
        ColVector v(3, vec_data);
        cout << "Vector v:" << endl << v << endl;
        v = tm * v;
        cout << "Result of tm * v = " << v << endl;
        if (!epsilonCheck(v(0), 70) ||
            !epsilonCheck(v(1), -184))
            return false;
    }

    // nthColumn
    {
        const double m_data[] = { 1, 2, 3,
                                  2, 1, 4,
                                  3, 4, 1 };
        Matrix m(3, 3, m_data);
        cout << "---" << endl
             << "Matrix m:" << endl << m;
        ColVector v = m.nthColumn(1);
        cout << "2nd column of m: " << v << endl;
        if (!epsilonCheck(v(0), 2) ||
            !epsilonCheck(v(1), 1) ||
            !epsilonCheck(v(2), 4))
            return false;
    }

    // nthRow
    {
        const double m_data[] = { 1, 2, 3,
                                  7, 8, 9,
                                  3, 4, 1 };
        Matrix m(3, 3, m_data);
        cout << "---" << endl
             << "Matrix m:" << endl << m;
        RowVector v = m.nthRow(1);
        cout << "2nd row of m: " << v << endl;
        if (!epsilonCheck(v(0), 7) ||
            !epsilonCheck(v(1), 8) ||
            !epsilonCheck(v(2), 9))
            return false;
    }

    // Eigenpairs
    {
        const double m_data[] = { 1, 2, 3,
                                  2, 1, 4,
                                  3, 4, 1 };
        Matrix m(3, 3, m_data);
        cout << "---" << endl
             << "Matrix m:" << endl << m;
        cout << "Eigenpairs: " << endl;
        Matrix::EigenPairs eigenp = m.eigenPairs();
        if (eigenp.size() != 3)
            return false;
        for (Matrix::EigenPairs::const_iterator it = eigenp.begin();
            it != eigenp.end(); ++it)
        {
            cout << it->first << " -> " << it->second << endl;
        }
        if (!epsilonCheck(eigenp.at(0).first, 7.07, 1e-2) ||
            !epsilonCheck(eigenp.at(1).first, -3.19, 1e-2) ||
            !epsilonCheck(eigenp.at(2).first, -0.89, 1e-2))
            return false;
    }

    // Multiplication with a transposed matrix
    {
        const double A_data[] = {   1, 2.5, 1.5,   -1, -2.5, -1.5,
                                  0.5,   0,   2, -0.5,    0,   -2 };
        Matrix A(2, 6, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        Matrix B = A.multWithTransposedMatrix(A);
        cout << "Result of A * A^T (method 1):" << endl << B;
        if (!epsilonCheck(B(0,0), 19) ||
            !epsilonCheck(B(0,1), 7) ||
            !epsilonCheck(B(1,0), 7) ||
            !epsilonCheck(B(1,1), 8.5))
            return false;

        A.multWithTransposedMatrix(A, &B);
        cout << "Result of A * A^T (method 2):" << endl << B;
        if (!epsilonCheck(B(0,0), 19) ||
            !epsilonCheck(B(0,1), 7) ||
            !epsilonCheck(B(1,0), 7) ||
            !epsilonCheck(B(1,1), 8.5))
            return false;

        A.multWithMatrix(A, &B, false, true, A.rows(), A.cols(), B.cols(), 
            0, 0, 0, 0, 0, 0);
        cout << "Result of A * A^T (method 3):" << endl << B;
        if (!epsilonCheck(B(0,0), 19) ||
            !epsilonCheck(B(0,1), 7) ||
            !epsilonCheck(B(1,0), 7) ||
            !epsilonCheck(B(1,1), 8.5))
            return false;
    }

    // Submatrix multiplication
    {
        cout << "---" << endl << "Submatrix and transposed multiplication" << endl;
        const double A_data[] = { 1, 4, 7, 1,
                                  2, 5, 8, 4,
                                  3, 6, 9, 7 };
        const double B_data[] = { 1, 1, 1,
                                  1, 2, 3,
                                  3, 4, 1,
                                  5, 6, 7 };
        const double X_data[] = { 1, 0, 3, 4,
                                  0, 2, 3, 4 };
        const double H_data[] = { 1, 0, 2, 5,
                                  0, 1, 0, 5 };
        const double W_data[] = { 1, 2,
                                  3, 4 };

        Matrix A(3, 4, A_data);
        Matrix B(4, 3, B_data);
        Matrix X(2, 4, X_data);
        Matrix H(2, 4, H_data);
        Matrix W(2, 2, W_data);
        Matrix C(3, 3);
        Matrix D(2, 2);
        Matrix E(2, 2);
        Matrix XHT(2, 2);
        Matrix WTX(2, 4);
        WTX.zero();
        A.multWithMatrix(B, &C);
        A.multWithMatrix(B, &C,
            false, false, 3, 3, 3,
            0, 0, 0, 0, 0, 0);
        A.multWithMatrix(B, &D,
            false, false,
            2, 2, 2,
            1, 2, 1, 0, 0, 0);
        A.multWithMatrix(B, &E,
            false, true,
            2, 1, 2,
            1, 3, 2, 2, 0, 0);
        X.multWithMatrix(H, &XHT, 
            false, true,
            X.rows(), X.cols() - 1, H.rows(),
            0, 1, 0, 0, 0, 0);
        W.multWithMatrix(X, &WTX,
            true, false,
            2, 2, 3,
            0, 0, 0, 1, 0, 1);
        cout << "C = " << endl << C << "D = " << endl << D << "E = " << endl << E;
        cout << "X = " << endl << X;
        cout << "H = " << endl << H;
        cout << "X*H(->1)^T = " << endl << XHT;
        cout << "W^T*X(<-1) = " << endl << WTX;
        //return false;

        const double C_corr_data[] = { 26, 37, 20,
                                       31, 44, 25,
                                       36, 51, 30 };
        const double D_corr_data[] = { 20, 32, 30, 46};
        const double E_corr_data[] = { 4, 28, 7, 49 };
        const double XHT_corr_data[] = { 8, 3, 10, 3 };
        const double WTX_corr_data[] = { 0, 6, 12, 16, 
                                         0, 8, 18, 24 };
        Matrix C_corr(3, 3, C_corr_data);
        Matrix D_corr(2, 2, D_corr_data);
        Matrix E_corr(2, 2, E_corr_data);
        Matrix XHT_corr(2, 2, XHT_corr_data);
        Matrix WTX_corr(2, 4, WTX_corr_data);
        if (C != C_corr || D != D_corr || E != E_corr || 
            XHT != XHT_corr || WTX != WTX_corr) 
        {
            return false;
        }
    }

    // Mean column vector
    {
        const double A_data[] = {   3, 2.5, 1.5,   -1, -2.5, -1.5,
                                  0.5,   -4,   2, -0.5,    0,   -2 };
        Matrix A(2, 6, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        const ColVector cv = A.meanColumnVector();
        cout << "Mean column vector = " << cv << endl;
        if (!epsilonCheck(cv.at(0), 0.3333, 1e-4) ||
            !epsilonCheck(cv.at(1), -0.6666, 1e-4))
            return false;
    }

    // Mean row vector
    {
        const double A_data[] = {   3, 2.5, 1.5,   -1, -2.5, -1.5,
                                  0.5,   -4,   2, -0.5,    0,   -2 };
        Matrix A(2, 6, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        const RowVector rv = A.meanRowVector();
        cout << "Mean row vector = " << rv << endl;
        if (!epsilonCheck(rv.at(0), 1.75, 1e-2) ||
            !epsilonCheck(rv.at(1), -0.75, 1e-2) ||
            !epsilonCheck(rv.at(2), 1.75, 1e-2) ||
            !epsilonCheck(rv.at(3), -0.75, 1e-2) ||
            !epsilonCheck(rv.at(4), -1.25, 1e-2) ||
            !epsilonCheck(rv.at(5), -1.75, 1e-2))
            return false;
    }

    // Variances
    {
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        ColVector varRows = A.varianceRows();
        RowVector varColumns = A.varianceColumns();
        cout << "---" << endl
             << "Matrix A:" << endl << A
             << "Variances over A's rows: " << varRows << endl
             << "Variances over A's columns: " << varColumns << endl;
        if (!epsilonCheck(varRows(0), 10.67, 1e-2) ||
            !epsilonCheck(varRows(1), 0.67, 1e-2) ||
            !epsilonCheck(varRows(2), 126.00, 1e-2) ||
            !epsilonCheck(varColumns(0), 169.56, 1e-2) ||
            !epsilonCheck(varColumns(1), 10.89, 1e-2) ||
            !epsilonCheck(varColumns(2), 34.89, 1e-2))
            return false;
    }

    // upToAndIncludingRow
    {
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        Matrix B = A.upToAndIncludingRow(1);
        cout << "Matrix B (= A.upToAndIncludingRow(2)):" << endl << B;
        if (B.rows() != 2 || B.cols() != 3)
            return false;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                if (A(i,j) != B(i,j))
                    return false;
            }
        }
    }

    // elementWiseDivision
    {
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        Matrix B(3, 3);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        A.elementWiseDivision(A, &B);
        cout << "Matrix B = A ./ A:" << endl << B;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (B(i,j) != 1)
                    return false;
            }
        }
    }

    // rowSum and colSum
    {
        const double A_data[] = {  3, -1,  7,  4,
                                   6,  7,  8,  -2,
                                 -23,  4, -5,  2 };
        Matrix A(3, 4, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        double cs = A.colSum(1);
        double rs = A.rowSum(1);
        cout << "Sum of column 1: " << cs << endl;
        cout << "Sum of row 1: " << rs << endl;
        if (cs != 10 || rs != 19)
            return false;

        rs = A.rowSum(1, 1, 2);
        cout << "Sum of row 1 (cols 1-2): " << rs << endl;
        if (rs != 15)
            return false;
    }

    // dotColCol and dotRowRow
    {
        const double A_data[] = {  3, -1,  7,
                                   6,  7,  8,
                                 -23,  4, -5 };
        Matrix A(3, 3, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        double dcc = Matrix::dotColCol(A, 1, A, 2);
        double drr = Matrix::dotRowRow(A, 1, A, 2);
        cout << "Dot-product of columns 1 and 2: " << dcc << endl;
        cout << "Dot-product of rows 1 and 2: " << drr << endl;
        if (dcc != 29 || drr != -150)
            return false;
    }

    // Silent operator() check
    // In case of errors this would fail during compile time
    {
    	Matrix a(5,5);
    	const Matrix b(3,3);
    	a(1,1) = b(2,2);
    }

    // Dump and read
    {
        const double A_data[] = {   3, 2.5, 1.5,   -1, -2.5, -1.5,
                                  0.5,   -4,   2, -0.5,    0,   -2 };
        Matrix A(3, 4, A_data);
        Poco::TemporaryFile tmpFile;
        A.dump(tmpFile.path());
        Matrix B(tmpFile.path());
        cout << "---" << endl
             << "Write matrix to file:" << endl << A
             << "Read matrix from file:" << endl << B;
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 4; ++j)
                if (A(i,j) != B(i,j)) return false;
    }

    // Inverse
    {
        const double A_data[] = {  1,    2, 3,
                                  -4,   -5, 6,
                                  -7, -0.5, 3 };
        Matrix A(3, 3, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        Matrix Ainv = A.inverse();
        cout << "Inverse of A:" << endl << Ainv;
        if (!epsilonCheck(Ainv(0,0),  0.07, 1e-2) ||
            !epsilonCheck(Ainv(1,0),  0.18, 1e-2) ||
            !epsilonCheck(Ainv(2,0),  0.19, 1e-2) ||
            !epsilonCheck(Ainv(0,1),  0.04, 1e-2) ||
            !epsilonCheck(Ainv(1,1), -0.14, 1e-2) ||
            !epsilonCheck(Ainv(2,1),  0.08, 1e-2) ||
            !epsilonCheck(Ainv(0,2), -0.16, 1e-2) ||
            !epsilonCheck(Ainv(1,2),  0.11, 1e-2) ||
            !epsilonCheck(Ainv(2,2), -0.02, 1e-2))
        {
            return false;
        }
    }

    // Moore-Penrose Pseudo-Inverse
    {
        const double A_data[] = { 22,  60,  76,
                                  14, -95, -44,
                                  16, -20,  24,
                                   9, -25,  65,
                                  99,  51,  86 };
        Matrix A(5, 3, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        Matrix Apinv = A.pseudoInverse();
        cout << "Moore-Penrose Pseudo-Inverse of A:" << endl << Apinv;
        if (!epsilonCheck(Apinv(0,0), -0.003, 1e-3) ||
            !epsilonCheck(Apinv(1,0),  0.001, 1e-3) ||
            !epsilonCheck(Apinv(2,0),  0.005, 1e-3) ||
            !epsilonCheck(Apinv(0,1),  0.006, 1e-3) ||
            !epsilonCheck(Apinv(1,1), -0.006, 1e-3) ||
            !epsilonCheck(Apinv(2,1), -0.002, 1e-3) ||
            !epsilonCheck(Apinv(0,2), -0.000, 1e-3) ||
            !epsilonCheck(Apinv(1,2), -0.003, 1e-3) ||
            !epsilonCheck(Apinv(2,2),  0.003, 1e-3) ||
            !epsilonCheck(Apinv(0,3), -0.007, 1e-3) ||
            !epsilonCheck(Apinv(1,3), -0.007, 1e-3) ||
            !epsilonCheck(Apinv(2,3),  0.011, 1e-3) ||
            !epsilonCheck(Apinv(0,4),  0.011, 1e-3) ||
            !epsilonCheck(Apinv(1,4),  0.002, 1e-3) ||
            !epsilonCheck(Apinv(2,4), -0.002, 1e-3))
        {
            return false;
        }
    }

    // Covariance
    {
        const double A_data[] = {   3, 2.5, 1.5,   -1, -2.5, -1.5,
                                  0.5,   -4,   2, -0.5,    0,   -2 };
        Matrix A(2, 6, A_data);
        cout << "---" << endl
             << "Matrix A:" << endl << A;
        Matrix C = A.covarianceMatrix();
        cout << "Covariance matrix of A:" << endl << C;
        if (!epsilonCheck(C(0,0),  5.27, 1e-2) ||
            !epsilonCheck(C(1,0), -0.13, 1e-2) ||
            !epsilonCheck(C(0,1), -0.13, 1e-2) ||
            !epsilonCheck(C(1,1),  4.37, 1e-2))
        {
            return false;
        }
    }
    return true;
}


} // namespace Testing
