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

#include "CNMFTest.h"
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


bool CNMFTest::performTest()
{
    //srand((unsigned int) time(NULL));

    const Elem xData[] = {
        0.5377, 1.3499, 0.6715, 0.8884, 0.1022,
        1.8339, 3.0349, 1.2075, 1.1471, 0.2414,
        2.2588, 0.7254, 0.7172, 1.0689, 0.3192,
        0.8622, 0.0631, 1.6302, 0.8095, 0.3129,
        0.3188, 0.7147, 0.4889, 2.9443, 0.8649,
        1.3077, 0.2050, 1.0347, 1.4384, 0.0301,
        0.4336, 0.1241, 0.7269, 0.3252, 0.1649,
        0.3426, 1.4897, 0.3034, 0.7549, 0.6277,
        3.5784, 1.4090, 0.2939, 1.3703, 1.0933,
        2.7694, 1.4172, 0.7873, 1.7115, 1.1093
    };
    Matrix x(10, 5, xData);
    cout << "10x5 matrix:" << endl;
    cout << x;
    cout << "---" << endl;

    const Elem wData[] = {
        0.8637, 1.0891, 0.6156, 1.4193, 1.1480, 0.8404, 2.1384, 2.9080, 0.3538, 0.0229,
        0.0774, 0.0326, 0.7481, 0.2916, 0.1049, 0.8880, 0.8396, 0.8252, 0.8236, 0.2620,
        1.2141, 0.5525, 0.1924, 0.1978, 0.7223, 0.1001, 1.3546, 1.3790, 1.5771, 1.7502,
        1.1135, 1.1006, 0.8886, 1.5877, 2.5855, 0.5445, 1.0722, 1.0582, 0.5080, 0.2857,
        0.0068, 1.5442, 0.7648, 0.8045, 0.6669, 0.3035, 0.9610, 0.4686, 0.2820, 0.8314,
        1.5326, 0.0859, 1.4023, 0.6966, 0.1873, 0.6003, 0.1240, 0.2725, 0.0335, 0.9792,
        0.7697, 1.4916, 1.4224, 0.8351, 0.0825, 0.4900, 1.4367, 1.0984, 1.3337, 1.1564,
        0.3714, 0.7423, 0.4882, 0.2437, 1.9330, 0.7394, 1.9609, 0.2779, 1.1275, 0.5336,
        0.2256, 1.0616, 0.1774, 0.2157, 0.4390, 1.7119, 0.1977, 0.7015, 0.3502, 2.0026,
        1.1174, 2.3505, 0.1961, 1.1658, 1.7947, 0.1941, 1.2078, 2.0518, 0.2991, 0.9642
    };
    Matrix wInit(10, 10, wData);

    // Matrix W after 1 iteration with continuity parameter 1
    const Elem w100Data[] = {
        0.0729, 0.1012, 0.0554, 0.1331, 0.1014, 0.0569, 0.1398, 0.2178, 0.0281, 0.0017,
        0.0325, 0.0153, 0.3456, 0.1914, 0.0642, 0.2769, 0.3348, 0.4074, 0.3262, 0.1292,
        0.1983, 0.0631, 0.0225, 0.0322, 0.1367, 0.0126, 0.1977, 0.2868, 0.2476, 0.2745,
        0.1470, 0.1557, 0.0984, 0.0841, 0.1910, 0.0557, 0.0807, 0.0948, 0.0607, 0.0303,
        0.0022, 0.2532, 0.1817, 0.1155, 0.1238, 0.1131, 0.2826, 0.1399, 0.0956, 0.1176,
        0.3753, 0.0132, 0.2143, 0.1106, 0.0402, 0.1036, 0.0200, 0.0724, 0.0076, 0.1655,
        0.0459, 0.0868, 0.0679, 0.0306, 0.0038, 0.0218, 0.0571, 0.0577, 0.0728, 0.0614,
        0.0347, 0.0689, 0.0539, 0.0335, 0.2371, 0.0849, 0.2496, 0.0311, 0.1143, 0.0575,
        0.0712, 0.1521, 0.0351, 0.0876, 0.2058, 0.4980, 0.0785, 0.3739, 0.1145, 0.7373,
        0.1951, 0.2620, 0.0260, 0.1946, 0.3458, 0.0372, 0.2560, 0.4797, 0.0551, 0.1760
    };
    Matrix wIter(10, 10, w100Data);

    const Elem hData[] = {
        0.5201, 0.2938, 1.3320, 1.3617, 0.1952,
        0.0200, 0.8479, 2.3299, 0.4550, 0.2176,
        0.0348, 1.1201, 1.4491, 0.8487, 0.3031,
        0.7982, 2.5260, 0.3335, 0.3349, 0.0230,
        1.0187, 1.6555, 0.3914, 0.5528, 0.0513,
        0.1332, 0.3075, 0.4517, 1.0391, 0.8261,
        0.7145, 1.2571, 0.1303, 1.1176, 1.5270,
        1.3514, 0.8655, 0.1837, 1.2607, 0.4669,
        0.2248, 0.1765, 0.4762, 0.6601, 0.2097,
        0.5890, 0.7914, 0.8620, 0.0679, 0.6252,
    };
    Matrix hInit(10, 5, hData);

    // Matrix H after 1 iteration with continuity parameter 1
    const Elem h100Data[] = {
        0.5277, 0.6290, 1.2392, 1.2358, 0.3941,
        0.0492, 1.2702, 1.8233, 1.1065, 0.2485,
        0.1197, 0.9846, 1.3278, 1.0720, 0.3484,
        1.5472, 1.9489, 0.9158, 0.3798, 0.0249,
        1.2048, 1.1656, 0.8049, 0.4635, 0.0801,
        0.2046, 0.3074, 0.6227, 0.7902, 0.5995,
        0.8592, 0.8362, 0.3377, 1.1762, 0.9734,
        1.2090, 0.9057, 0.4086, 0.8553, 0.5399,
        0.1724, 0.3421, 0.5071, 0.4923, 0.3308,
        0.6972, 0.8409, 0.6026, 0.2571, 0.2429
    };
    Matrix hIter(10, 5, h100Data);

    // This sanity check tests whether a continuous NMF with the continuity
    // weight set to zero equals normal NMF. This test is necessary since
    // implementation might select different algorithms for both tasks.
    // In this test, of course, the same initialization has to be chosen
    // for both variants.

    cout << "Performing Continuous NMF sanity check" << endl << endl;

    {
        cout << "KL divergence" << endl;
        nmf::Deconvolver d(x, 10, 1);
        Matrix w(10, 10, nmf::gaussianRandomGenerator);
        Matrix h(10, 5, nmf::gaussianRandomGenerator);
        d.setW(0, w);
        d.setH(h);
        d.decompose(Deconvolver::KLDivergence, 1, 0.0, Deconvolver::NoSparsity, true);
        d.computeApprox();
        Matrix wh1(d.getApprox());
        d.setW(0, w);
        d.setH(h);
        d.decompose(Deconvolver::KLDivergence, 1, 0.0, Deconvolver::NoSparsity, false);
        d.computeApprox();
        Matrix wh2(d.getApprox());
        cout << "1 iteration of continuous NMF (param = 0)" << endl;
        cout << wh1;
        cout << "1 iteration of NMF" << endl;
        cout << wh2;
        if (!epsilonCheck(wh1, wh2, 1e-2)) {
            return false;
        }
    }

    // Numerical soundness test for continuity parameter = 1.

    {
        double continuity = 1;

        cout << endl << "---" << endl;
        cout << "Performing Continuous NMF using KL divergence" << endl;
        cout << "Continuity parameter set to " << continuity << endl;

        nmf::Deconvolver d(x, 10, 1);
        d.setW(0, wInit);
        d.setH(hInit);
        Matrix c(10, 5);
        for (unsigned int j = 0; j < c.cols(); ++j) {
            for (unsigned int i = 0; i < c.rows(); ++i) {
                c(i, j) = continuity;
            }
        }
        d.setContinuity(c);

        d.decompose(nmf::Deconvolver::KLDivergence, 1, 0.0, Deconvolver::NoSparsity, true);
        d.computeApprox();
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;

        cout << "W = " << endl;
        cout << d.getW(0) << endl;
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "WH = " << endl;
        Matrix l(d.getApprox());
        cout << l << endl;
        
        // Reconstruction seems not to be feasible for these matrix
        // dimensions, hence just check numerical soundness.
        if (!epsilonCheck(d.getW(0), wIter, 1e-3)) {
            cout << "Error in W" << endl;
            return false;
        }

        if (!epsilonCheck(d.getH(), hIter, 1e-3)) {
            cout << "Error in H" << endl;
            return false;
        }
    }

    {
        double continuity = 0.01;

        cout << endl << "---" << endl;
        cout << "Performing Sparse Continuous NMF using ED" << endl;
        cout << "Sparsity and continuity parameter set to " << continuity << endl;

        nmf::Deconvolver d(x, 10, 1);
        d.setW(0, wInit);
        d.setH(hInit);
        Matrix c(10, 5);
        for (unsigned int j = 0; j < c.cols(); ++j) {
            for (unsigned int i = 0; i < c.rows(); ++i) {
                c(i, j) = continuity;
            }
        }
        d.setContinuity(c);
        d.setSparsity(c);

        d.factorizeNMDBeta(1000, 1e-5, 2, Deconvolver::NormalizedL1Norm, true);
        d.computeApprox();
        cout << "absolute error: " << d.absoluteError() << endl;
        cout << "relative error: " << d.relativeError() << endl;
        cout << endl;

        cout << "W = " << endl;
        cout << d.getW(0) << endl;
        cout << "H = " << endl;
        cout << d.getH() << endl;
        cout << "WH = " << endl;
        Matrix l(d.getApprox());
        cout << l << endl;
        
        // Reconstruction seems not to be feasible for these matrix
        // dimensions, don't check anything here.
    }

    return true;
}


} // namespace Testing
