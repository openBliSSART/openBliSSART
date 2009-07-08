//
// $Id: CorrelationFeatureSelector.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <blissart/CorrelationFeatureSelector.h>
#include <cmath>


using namespace std;


namespace blissart {


FeatureSelector::FeatureScoreMap 
CorrelationFeatureSelector::rateFeatures(const DataSet& dataSet)
{
    map<FeatureDescriptor, LabelledVec> featureSamples;
    FeatureScoreMap result;

    // For each feature, build a pair of the feature's values in the
    // DataDescriptors and the corresponding class labels.
    for (DataSet::const_iterator itr = dataSet.begin();
        itr != dataSet.end(); ++itr)
    {
        for (DataPoint::ComponentMap::const_iterator compItr = itr->components.begin();
            compItr != itr->components.end(); ++compItr)
        {
            featureSamples[compItr->first].first.push_back(compItr->second);
            featureSamples[compItr->first].second.push_back(itr->classLabel);
        }
    }

    // Measure the correlation between values and class labels for each feature.
    for (map<FeatureDescriptor, LabelledVec>::const_iterator itr = featureSamples.begin();
        itr != featureSamples.end(); ++itr)
    {
        result[itr->first] = correlation(itr->second.first, itr->second.second);
    }

    return result;
}


double CorrelationFeatureSelector::correlation(const vector<double>& x, const vector<double>& y)
{
    double result = 0.0;
    vector<double>::const_iterator xItr = x.begin();
    vector<double>::const_iterator yItr = y.begin();
    for (; xItr != x.end() && yItr != y.end(); ++xItr, ++yItr)
    {
        result += (*xItr) * (*yItr);
    }
    return (result - x.size() * mean(x) * mean(y)) / ((x.size() - 1) * sqrt(variance(x) * variance(y)));
}


bool CorrelationFeatureSelector::lowerScoreIsHigherRank()
{
    return false;
}


bool CorrelationFeatureSelector::rankByAbsoluteScore()
{
    return true;
}


} // namespace blissart
