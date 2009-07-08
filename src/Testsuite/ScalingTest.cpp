//
// $Id: ScalingTest.cpp 855 2009-06-09 16:15:50Z alex $
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

#include "ScalingTest.h"
#include <blissart/DataSet.h>
#include <iostream>
#include <vector>


using namespace std;
using namespace blissart;


namespace Testing {


void outputDataSet(const DataSet& set)
{
    for (DataSet::const_iterator dItr = set.begin();
        dItr != set.end(); ++dItr)
    {
        for (DataPoint::ComponentMap::const_iterator cItr = dItr->components.begin();
            cItr != dItr->components.end(); ++cItr)
        {
            cout << cItr->second << " ";
        }
        cout << endl;
    }
}


bool ScalingTest::performTest()
{
    DataSet ds1, ds2;
    DataPoint dp;

    FeatureDescriptor fd1("stddev", DataDescriptor::Spectrum);
    FeatureDescriptor fd2("stddev", DataDescriptor::Gains);

    dp.components[fd1] = 5;
    dp.components[fd2] = 8;
    ds1.push_back(dp);

    dp.components[fd1] = -8;
    dp.components[fd2] = 6;
    ds1.push_back(dp);

    dp.components[fd1] = 4;
    dp.components[fd2] = 5;
    ds1.push_back(dp);

    dp.components[fd1] = -15;
    dp.components[fd2] = 10;
    ds1.push_back(dp);

    DataSet dsCopy = ds1;
    linearScaleMinMax(dsCopy);
    outputDataSet(dsCopy);
    cout << endl;

    dsCopy = ds1;
    linearScaleMinMax(dsCopy, 0, 1);
    outputDataSet(dsCopy);
    cout << endl;

    dp.components[fd1] = 10;
    dp.components[fd2] = 0;
    ds2.push_back(dp);

    vector<DataSet> dsVec(2);
    dsVec[0] = ds1;
    dsVec[1] = ds2;
    linearScaleMinMax(dsVec);
    outputDataSet(dsVec[0]);
    cout << endl;
    outputDataSet(dsVec[1]);
    cout << endl;

    DataSet ds3;
    DataPoint dp2;
    dp2.components[fd1] = 1;
    ds3.push_back(dp2);
    dp2.components[fd1] = 2;
    ds3.push_back(dp2);
    dp2.components[fd1] = 3;
    ds3.push_back(dp2);
    dp2.components[fd1] = 4;
    ds3.push_back(dp2);
    dp2.components[fd1] = 5;
    ds3.push_back(dp2);
    linearScaleMuSigma(ds3, 1, 2);
    outputDataSet(ds3);
    cout << endl;

    return true;
}


} // namespace Testing
