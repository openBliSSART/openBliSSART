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


#include "VectorTest.h"
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>

#include <Poco/TemporaryFile.h>

#include <iostream>
#include <ctime>


using namespace blissart;
using namespace blissart::linalg;
using namespace std;


namespace Testing {


bool VectorTest::performTest()
{
    srand((unsigned)time(NULL));

    const Elem cv_data[] = {  1, 2, 3 };
    const Elem rv_data[] = { -4, 0, 8 };
    ColVector cv(3, cv_data);
    RowVector rv(3, rv_data);
    cout << "Vector cv: " << cv << endl
         << "Vector rv: " << rv << endl;

    // Shift
    {
        const Elem cv_left[] = { 2, 3, 0 };
        const Elem cv_right[] = { 0, 1, 2 };
        ColVector cvl(cv);
        cvl.shiftLeft();
        cout << "Vector rv shifted left: " << cvl << endl;
        if (ColVector(3, cv_left) != cvl) return false;
        ColVector cvr(cv);
        cvr.shiftRight();
        cout << "Vector rv shifted right: " << cvr << endl;
        if (ColVector(3, cv_right) != cvr) return false;
    }


    // rv + rv
    {
        RowVector result = rv + rv;
        cout << "rv + rv = " << result << endl;
        for (unsigned int i = 0; i < rv.dim(); i++) {
            if (!epsilonCheck(result(i), rv(i) + rv(i)))
                return false;
        }
    }

    // rv - rv
    {
        RowVector result = rv - rv;
        cout << "rv - rv = " << result << endl;
        for (unsigned int i = 0; i < rv.dim(); i++) {
            if (!epsilonCheck(result(i), rv(i) - rv(i)))
                return false;
        }
    }

    // cv + cv
    {
        ColVector result = cv + cv;
        cout << "cv + cv = " << result << endl;
        for (unsigned int i = 0; i < cv.dim(); i++) {
            if (!epsilonCheck(result(i), cv(i) + cv(i)))
                return false;
        }
    }

    // cv - cv
    {
        ColVector result = cv - cv;
        cout << "cv - cv = " << result << endl;
        for (unsigned int i = 0; i < cv.dim(); i++) {
            if (!epsilonCheck(result(i), cv(i) - cv(i)))
                return false;
        }
    }

    // Operators += and -=
    {
        ColVector cvc(cv);
        cout << "cvc = " << cvc << endl;

        cvc += cv;
        cout << "cvc += cv => " << cvc << endl;
        if (!epsilonCheck(cvc(0), 2) ||
            !epsilonCheck(cvc(1), 4) ||
            !epsilonCheck(cvc(2), 6))
            return false;

        cvc -= cv;
        cout << "cvc -= cv => " << cvc << endl;
        if (!epsilonCheck(cvc(0), 1) ||
            !epsilonCheck(cvc(1), 2) ||
            !epsilonCheck(cvc(2), 3))
            return false;

        RowVector rvc(rv); // -4 0 8
        cout << "rvc = " << rvc << endl;

        rvc += rv;
        cout << "rvc += rv => " << rvc << endl;
        if (!epsilonCheck(rvc(0), -8) ||
            !epsilonCheck(rvc(1), 0) ||
            !epsilonCheck(rvc(2), 16))
            return false;

        rvc -= rv;
        cout << "rvc -= rv => " << rvc << endl;
        if (!epsilonCheck(rvc(0), -4) ||
            !epsilonCheck(rvc(1), 0) ||
            !epsilonCheck(rvc(2), 8))
            return false;
    }

    // Operator *=
    {
        ColVector cvc(cv);
        cout << "cvc = " << cvc << endl;
        cvc *= 5;
        cout << "cvc *= 5 => " << cvc << endl;
        if (!epsilonCheck(cvc(0), 5) ||
            !epsilonCheck(cvc(1), 10) ||
            !epsilonCheck(cvc(2), 15))
            return false;
    }

    // rv * cv
    {
        const Elem inner_prod = rv * cv;
        cout << "rv * cv = " << inner_prod << endl;
        if (!epsilonCheck(inner_prod, 20))
            return false;
    }

    // cv * rv
    {
        const Matrix cvrv(cv * rv);
        cout << "cv * rv = " << endl << cvrv;
        const Elem correct_cvrv_data[] = {  -4, 0,  8,
                                              -8, 0, 16,
                                             -12, 0, 24 };
        const Matrix correct_cvrv(3, 3, correct_cvrv_data);
        if (correct_cvrv != cvrv)
            return false;
    }

    // cv * rv larger test
    {
        int n_tests = 100;
        for (int k = 0; k < n_tests; ++k) {
            cout << "cv * rv #" << k << endl;
            const int m = 25;
            const int n = 500;
            double* data1 = new double[m];
            double* data2 = new double[n];
            for (int i = 0; i < m; ++i)
                data1[i] = generators::random(i);
            for (int i = 0; i < n; ++i)
                data2[i] = generators::random(i);
            ColVector randomCv(m, data1);
            RowVector randomRv(n, data2);
            Matrix cvAsMatrix(m, 1, data1);
            Matrix rvAsMatrix(1, n, data2);
            Matrix ref(m, n);
            cvAsMatrix.multWithMatrix(rvAsMatrix, &ref);
            Matrix res(randomCv * randomRv);
            if (ref != res) 
                return false;
            delete[] data2;
            delete[] data1;
        }
    }

    // cv * 5
    {
        ColVector tmp = cv * 5;
        cout << "cv * 5 = " << tmp << endl;
        if (!epsilonCheck(tmp(0), 5) ||
            !epsilonCheck(tmp(1), 10) ||
            !epsilonCheck(tmp(2), 15))
            return false;
    }

    // 4 * cv
    {
        ColVector tmp = 4 * cv;
        cout << "4 * cv = " << tmp << endl;
        if (!epsilonCheck(tmp(0), 4) ||
            !epsilonCheck(tmp(1), 8) ||
            !epsilonCheck(tmp(2), 12))
            return false;
    }

    // rv * 5
    {
        RowVector tmp = rv * 5;
        cout << "rv * 5 = " << tmp << endl;
        if (!epsilonCheck(tmp(0), -20) ||
            !epsilonCheck(tmp(1), 0) ||
            !epsilonCheck(tmp(2), 40))
            return false;
    }

    // 4 * rv
    {
        RowVector tmp = 4 * rv;
        cout << "4 * rv = " << tmp << endl;
        if (!epsilonCheck(tmp(0), -16) ||
            !epsilonCheck(tmp(1), 0) ||
            !epsilonCheck(tmp(2), 32))
            return false;
    }

    // Maximum, minimum
    const Elem mv_data[] = { -16, -1, 4, 8 };
    RowVector mv(4, mv_data);
    cout << "mv = " << mv << endl;

    Elem m = mv.minimum();
    cout << "Minimum of mv:            " << m << endl;
    if (m != -16)
        return false;
    
    m = mv.minimum(true);
    cout << "Minimum of mv (absolute): " << m << endl;
    if (m != -1)
        return false;

    m = mv.maximum();
    cout << "Maximum of mv:            " << m << endl;
    if (m != 8)
        return false;
    
    m = mv.maximum(true);
    cout << "Maximum of mv (absolute): " << m << endl;
    if (m != -16)
        return false;

    // Length
    cout << "Length of rv: " << rv.length() << endl;
    if (!epsilonCheck(rv.length(), 8.94427191))
        return false;
    
    // Angle between cv and rv
    cout << "Angle between cv and rv: " << Vector::angle(cv, rv) << endl;
    if (!epsilonCheck(Vector::angle(cv, rv), 0.930274014115))
        return false;
    
    // Random vector of unit length
    ColVector randcv(3, generators::random);
    cout << "Random vector: " << randcv << endl;

    // Silent operator() check
    // In case of errors this would fail during compile time
    {
    	ColVector a(5);
    	const ColVector b(3);
    	a(1) = b(2);
    }
    
    // Dump and read
    {
        const Elem v_data[] = {   3, 2.5, 1.5,   -1, -2.5, -1.5 };
        RowVector rv(6, v_data);
        ColVector cv(6, v_data);
        Poco::TemporaryFile rvTmpFile, cvTmpFile;
        rv.dump(rvTmpFile.path());
        cv.dump(cvTmpFile.path());
        RowVector rv2(rvTmpFile.path());
        ColVector cv2(cvTmpFile.path());
        cout << "---" << endl
             << "rv = " << rv << endl
             << "cv = " << cv << endl
             << "rv from file: " << rv2 << endl
             << "cv from file: " << cv2 << endl;
        for (unsigned int i = 0; i < 6; ++i)
            if (rv(i) != rv2(i) || cv(i) != cv2(i))
                return false;
    }
    
    return true;
}


} // namespace Testing
