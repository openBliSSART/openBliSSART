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


#include "FeatureExtractionTest.h"

#include <blissart/feature/misc.h>
#include <blissart/feature/peak.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>
#include <blissart/FeatureDescriptor.h>
#include <blissart/FeatureExtractor.h>

#include <iostream>
#include <cmath>
#include <map>


using namespace std;
using namespace blissart;
using namespace blissart::feature;
using namespace blissart::linalg;


namespace Testing {


FeatureExtractionTest::FeatureExtractionTest()
{
}


bool FeatureExtractionTest::performTest()
{
    cout << "Testing FeatureDescriptor class as key type: ";
    FeatureDescriptor fd("pf", DataDescriptor::Gains);
    map<FeatureDescriptor, int> fdMap;
    fdMap[fd] = 5;
    cout << "5 = " << fdMap[fd] << endl;
    if (fdMap[fd] != 5)
        return false;

    const double data[] = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1 };
    RowVector v(16, data);

    const double data2[] = { 0.3, 0.5, 0.2, 0.7 };
    RowVector w(4, data2);

    cout << "v = " << v << endl;
    cout << "w = " << w << endl;

    double m = mean(v);
    cout << "Mean of v = " << m << endl;
    if (m != 0.625)
        return false;

    double s = stddev(v);
    cout << "Sample standard deviation of v = " << s << endl;
    if (!epsilonCheck(s, 0.5, 1e-3))
        return false;

    double sk = skewness(v);
    cout << "Sample skewness of v = " << sk << endl;
    if (!epsilonCheck(sk, -0.516, 1e-3))
        return false;

    double k = kurtosis(v);
    cout << "Sample kurtosis of v = " << k << endl;
    if (!epsilonCheck(k, -1.733, 1e-3))
        return false;

    double a = autocorrelation(v, 0);
    cout << "Autocorrelation of v (delay = 0): " << a << endl;
    if (a != 0.9375)
        return false;

    a = autocorrelation(v, 1);
    cout << "Autocorrelation of v (delay = 1): " << a << endl;
    if (!epsilonCheck(a, -0.0041, 0.001))
        return false;

    a = averagePeakLength(v);
    cout << "Average peak length in v: " << a << endl;
    if (a != 2.5)
        return false;

    double f = peakFluctuation(v);
    cout << "Peak fluctuation in v: " << f << endl;
    if (!epsilonCheck(f, sqrt(5.0 / 3.0), 1e-5))
        return false;

    double c = centroid(w, 100.0);
    cout << "Spectral centroid of w (base = 100 Hz): " << c << endl;
    if (!epsilonCheck(c, 44.12, 1e-2))
        return false;

    double freq[] = { 0.0, 25.0, 50.0, 75.0 };
    c = centroid(w, RowVector(4, freq));
    cout << "Spectral centroid of w (given frequencies): " << c << endl;
    if (!epsilonCheck(c, 44.12, 1e-2))
        return false;

    double r = rolloff(v, 100);
    cout << "Roll off (95%) of v: " << r << endl;
    if (!epsilonCheck(r, 93.75, 1e-2))
        return false;

    r = rolloff(w, 100);
    cout << "Roll off (95%) of w: " << r << endl;
    if (r != 75)
        return false;

    double sfm = spectralFlatness(w);
    cout << "Spectral flatness of w = " << sfm << endl;
    if (!epsilonCheck(0.666, sfm, 1e-3))
        return false;

    const double data3[] =     { 2, 3, 4, 5, 3, 2, 0, 1, 2, -1, -5, -3, -2, -3, 0, 1 };
    const double data3_max[] = { 2, 0, 0, 5, 0, 0, 0, 0, 2,  0,  0,  0, -2,  0, 0, 1 };
    RowVector u(16, data3);
    RowVector umax(16, data3_max);
    cout << "u = " << u << endl;
    RowVector maxima = findLocalMaxima(u);
    cout << "maxima of u = " << maxima << endl;
    if (maxima != umax)
        return false;

    const double data4[] = { 0, 1, 2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2, 1, 0 };
    RowVector x(17, data4);
    double nl = noiseLikeness(x);
    cout << "x = " << x << endl;
    cout << "Noise-likeness of x (sigma = 1.0): " << nl << endl;
    if (!epsilonCheck(nl, 0.87, 1e-2))
        return false;

    double pc = percussiveness(x, 3);
    cout << "percussiveness of x (length = 3): " << pc << endl;
    if (!epsilonCheck(pc, 0.70, 1e-2))
        return false;

    double diss = spectralDissonance(w, 100.0);
    cout << "dissonance of w: " << diss << endl;

    cout << endl;
    cout << "Testing FeatureExtractor on magnitude matrix" << endl;
    Matrix sp(512, 9, generators::random);
    FeatureExtractor fex;
    fex.setSampleFrequency(12500.0);
    FeatureExtractor::FeatureMap fm =
        fex.extract(DataDescriptor::MagnitudeMatrix, sp);
    for (FeatureExtractor::FeatureMap::const_iterator itr =
         fm.begin(); itr != fm.end(); ++itr)
    {
        cout << itr->first.toString() << ": " << itr->second << endl;
    }

    return true;

}


} // namespace Testing

