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


#include "MTrTest.h"
#include <blissart/transforms/PowerTransform.h>
#include <blissart/transforms/MelFilterTransform.h>
#include <blissart/transforms/SlidingWindowTransform.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>
#include <Poco/Util/Application.h>
#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/SharedPtr.h>
#include <iostream>


using namespace blissart;
using blissart::linalg::Matrix;
using namespace std;


namespace Testing {


bool MTrTest::performTest()
{
    {
        // Test PowerTransform and SlidingWindowTransform for some small 
        // matrices.
        const double s_data[] = { 2, 1.5, 0.5, 0.2, 1,
                                  4.5, 2.5, 0.3, 0.7, 1.2,
                                  0.6, 0.4, 3, 5, 6 };
        Matrix s(3, 5, s_data);
        Matrix sOrig(s);
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

        cout << "Applying power transform: " << endl;
        cout << s << endl;

        if (!epsilonCheck(s, sSq)) 
            return false;

        transforms::SlidingWindowTransform st;
        Poco::SharedPtr<Matrix> slRes = st.transform(&s);

        cout << endl;
        cout << "Applying sliding window transform: " << endl;
        cout << *slRes << endl;

        if (!epsilonCheck(*slRes, sSliding)) {
            return false;
        }

        Poco::SharedPtr<Matrix> slRev = st.inverseTransform(slRes);
        cout << "Reverting sliding window transform:" << endl;
        cout << *slRev << endl;
        if (!epsilonCheck(*slRev, sSq)) 
            return false;

        pt.inverseTransform(slRev);
        cout << "Reverting power transform:" << endl;
        cout << *slRev << endl;
        if (!epsilonCheck(*slRev, sOrig)) 
            return false;
    }

    {
        cout << endl;
        cout << "SlidingWindowTransform tests ..." << endl << endl;

        // Test SlidingWindowTransform for different parameters on bigger
        // random matrices.
        Matrix r(10, 49, blissart::linalg::generators::random);
        const int framesizes[] = { 1, 2, 5, 10, 20 };
        const int nFramesizes = 5;
        const int framerates[] = { 1, 2, 5, 10 };
        const int nFramerates = 4;
        Poco::Util::LayeredConfiguration& cfg 
            = Poco::Util::Application::instance().config();

        for (int fs = 0; fs < nFramesizes; ++fs) {
            for (int fr = 0; fr < nFramerates; ++fr) {
                // framesize must be at least the frame rate
                if (fr > fs) continue;

                cfg.setInt(
                    "blissart.fft.transformations.slidingWindow.frameSize", 
                    framesizes[fs]);
                cfg.setInt(
                    "blissart.fft.transformations.slidingWindow.frameRate", 
                    framerates[fr]);
                transforms::SlidingWindowTransform st;

                cout << "Framesize = " << framesizes[fs]
                     << " Framerate = " << framerates[fr] 
                     << " ... ";

                Poco::SharedPtr<Matrix> slRes = st.transform(&r);
                Poco::SharedPtr<Matrix> slRev;

                try {
                    slRev = st.inverseTransform(slRes);
                }
                catch (const Poco::Exception& exc) {
                    cout << " failed! Reason: " << exc.displayText() << endl;
                }
                catch (const std::exception& exc) {
                    cout << " failed! Reason: " << exc.what() << endl;
                }

                if (slRev->rows() != r.rows()) {
                    cout << " failed (row number mismatch: orig = " 
                         << r.rows() << " new = " << slRev->rows() << endl;
                    return false;
                }

                // Number of cols might differ, so we can't use our
                // fancy overloaded version of epsilonCheck().
                for (unsigned int j = 0; j < slRev->cols(); ++j) {
                    for (unsigned int i = 0; i < slRev->rows(); ++i) {
                        if (!epsilonCheck(r(i, j), (*slRev)(i, j))) {
                            cout << " failed!" << endl;
                            cout << "Original matrix:" << endl 
                                 << r << endl;
                            cout << "Transformed matrix:" << endl 
                                 << *slRes << endl;
                            cout << "Reverted matrix:" << endl 
                                 << *slRev << endl;
                            cout << "Position (" << i << "," << j << "): "
                                 << "orig = " << r(i, j) 
                                 << "new = " << (*slRev)(i, j) << endl;
                            return false;
                        }
                    }
                }

                cout << "OK!" << endl;
            }
        }
    }

    return true;
}


} // namespace Testing
