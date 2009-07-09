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


#include "FeatureSelectionTest.h"
#include <blissart/CorrelationFeatureSelector.h>
#include <blissart/AnovaFeatureSelector.h>
#include <blissart/FeatureSet.h>
#include <iostream>
#include <vector>


using namespace blissart;
using namespace std;


namespace Testing {


bool outputAndCheck(const DataSet& dataSet, const FeatureSet& featureSet)
{
    int i = 0;
    bool ok = true;
    for (DataSet::const_iterator dItr = dataSet.begin(); dItr != dataSet.end();
        ++dItr, ++i)
    {
        cout << "DataPoint #" << i << " has " << dItr->components.size()
             << " features: ";
        for (DataPoint::ComponentMap::const_iterator cItr = 
            dItr->components.begin(); cItr != dItr->components.end(); ++cItr)
        {
            cout << cItr->first.toString() << " ";
            if (!featureSet.has(cItr->first)) {
                ok = false;
            }
        }
        cout << endl;
        if (dItr->components.size() != featureSet.size())
            ok = false;
    }
    return ok;
}


void outputFeatureSet(const FeatureSelector::FeatureScoreMap& fs)
{
    for (FeatureSelector::FeatureScoreMap::const_iterator itr = fs.begin();
        itr != fs.end(); ++itr)
    {
        cout << itr->first.toString() << ": " << itr->second << endl;
    }
}


bool FeatureSelectionTest::performTest()
{
    DataSet dataSet;
    DataPoint dp;

    FeatureDescriptor f1("mfcc", DataDescriptor::Spectrum, 0);
    FeatureDescriptor f2("mfcc", DataDescriptor::Spectrum, 1);
    FeatureDescriptor f3("mfcc", DataDescriptor::Spectrum, 2);

    // 2 instances of class 1

    dp.components[f1] = 0.7;
    dp.components[f2] = 10.0;
    dp.components[f3] = -3.0;
    dp.classLabel = 1;
    dataSet.push_back(dp);

    dp.components[f1] = 0.3;
    dp.components[f2] = 11.0;
    dp.components[f3] = -2.7;
    dataSet.push_back(dp);

    // 3 instances of class 2

    dp.components[f1] = 0.8;
    dp.components[f2] = 1.0;
    dp.components[f3] = 5.1;
    dp.classLabel = 2;
    dataSet.push_back(dp);

    dp.components[f1] = 0.6;
    dp.components[f2] = 1.2;
    dp.components[f3] = 6.2;
    dataSet.push_back(dp);

    dp.components[f1] = 0.4;
    dp.components[f2] = 1.1;
    dp.components[f3] = 4.8;
    dataSet.push_back(dp);

    FeatureSet correctFeatureSet;
    correctFeatureSet.add(f2);
    correctFeatureSet.add(f3);

    cout << "Testing AnovaFeatureSelector" << endl;
    DataSet dataSetCopy = dataSet;
    AnovaFeatureSelector afs;
    FeatureSelector::FeatureScoreMap filteredFeatures = 
        afs.filterDataSet(dataSetCopy, 2);
    if (!outputAndCheck(dataSetCopy, correctFeatureSet))
        return false;

    cout << "Features selected:" << endl;
    outputFeatureSet(filteredFeatures);
    cout << endl;

    cout << "Testing CorrelationFeatureSelector" << endl;
    dataSetCopy = dataSet;
    CorrelationFeatureSelector cfs;
    filteredFeatures = cfs.filterDataSet(dataSetCopy, 2);
    outputAndCheck(dataSetCopy, correctFeatureSet);

    cout << "Features selected:" << endl;
    outputFeatureSet(filteredFeatures);

    return true;
}


} // namespace Testing
