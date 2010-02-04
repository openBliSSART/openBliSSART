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


#include "MTrTest.h"
#include <blissart/transforms/PowerTransform.h>
#include <blissart/transforms/MelFilterTransform.h>
#include <blissart/transforms/SlidingWindowTransform.h>
#include <blissart/linalg/Matrix.h>
#include <Poco/Util/Application.h>
#include <Poco/Util/LayeredConfiguration.h>
#include <iostream>


using namespace blissart;
using blissart::linalg::Matrix;
using namespace std;


namespace Testing {


bool MTrTest::performTest()
{
    const double s_data[] = { 2, 1.5, 0.5, 0.2, 1,
                              4.5, 2.5, 0.3, 0.7, 1.2,
                              0.6, 0.4, 3, 5, 6 };
    Matrix s(3, 5, s_data);
    cout << "S = " << endl;
    cout << s << endl;

    const double s_data_Sq[] = { 4, 2.25, 0.25, 0.04, 1,
                                 20.25, 6.25, 0.09, 0.49, 1.44,
                                 0.36, 0.16, 9, 25, 36 };
    Matrix sSq(3, 5, s_data_Sq);

    Poco::Util::LayeredConfiguration& cfg 
        = Poco::Util::Application::instance().config();
    cfg.setInt("blissart.fft.transformations.slidingWindow.frameSize", 2);

    const double s_data_sliding[] = { 4,     2.25, 0.25, 0.04,
                                      20.25, 6.25, 0.09, 0.49,
                                      0.36,  0.16,   9,  25,
                                      2.25,  0.25, 0.04, 1,
                                      6.25,  0.09, 0.49, 1.44,
                                      0.16,  9,    25,   36 };
    Matrix sSliding(6, 4, s_data_sliding);

    transforms::PowerTransform pt;
    pt.transform(&s);

    cout << "Power: " << endl;
    cout << s << endl;

    if (!epsilonCheck(s, sSq)) 
        return false;

    transforms::SlidingWindowTransform st;
    Matrix* slRes = st.transform(&s);

    cout << endl;
    cout << "Sliding window: " << endl;
    cout << *slRes << endl;

    if (!epsilonCheck(*slRes, sSliding)) {
        return false;
    }

    return true;
}


} // namespace Testing
