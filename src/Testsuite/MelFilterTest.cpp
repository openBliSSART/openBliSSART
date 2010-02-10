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


#include "MelFilterTest.h"
#include <blissart/audio/MelFilter.h>
#include <blissart/linalg/Matrix.h>
#include <Poco/SharedPtr.h>
#include <iostream>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using namespace blissart::audio;


namespace Testing {


bool MelFilterTest::performTest()
{
    // Create a constant spectrum of ones
    const unsigned int n = 200;
    MelFilter mf(5, 44100, 0, 0);
    Matrix s(n, 1);
    for (unsigned int i = 0; i < s.rows(); ++i)
        s(i, 0) = 1.0;

    Matrix s2(n, 1);

    // Compute Mel spectrum
    cout << "Meling ..." << endl;
    Poco::SharedPtr<Matrix> mel = mf.melSpectrum(s);

    // This spectrum should be resynthesized perfectly except for the
    // energy coefficient, which is ignored by the Mel filter.
    cout << "Unmeling ..." << endl;
    mf.synth(*mel, s2);
    s2(0, 0) = 1.0;
    if (!epsilonCheck(s, s2, 1e-2)) {
        cout << "Synth failed: " << endl;
        cout << s2;
        return false;
    }

    cout << "OK!" << endl;
    return true;
}


} // namespace Testing
