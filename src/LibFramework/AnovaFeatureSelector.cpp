//
// $Id: AnovaFeatureSelector.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <blissart/AnovaFeatureSelector.h>
#include <cmath>


using namespace std;


namespace blissart {


FeatureSelector::FeatureScoreMap AnovaFeatureSelector::rateFeatures(const DataSet& dataSet)
{
    map<FeatureDescriptor, SampleMap> featureSamples;
    FeatureScoreMap result;

    // For each feature, build a set of samples that correspond to each class.
    for (DataSet::const_iterator itr = dataSet.begin(); 
        itr != dataSet.end(); ++itr)
    {
        for (DataPoint::ComponentMap::const_iterator compItr = itr->components.begin();
            compItr != itr->components.end(); ++compItr)
        {
            featureSamples[compItr->first][itr->classLabel].push_back(compItr->second);
        }
    }

    // Calculate the t-test score for each feature.
    // We do not use the F-test because we do not assume variances in the
    // samples to be equal. Therefore, we can only handle 2 classes.
    for (map<FeatureDescriptor, SampleMap>::const_iterator itr = featureSamples.begin();
        itr != featureSamples.end(); ++itr)
    {
        if (itr->second.size() > 2)
            throw Poco::NotImplementedException("Anova for > 2 groups not implemented");
        else if (itr->second.size() < 2)
            throw Poco::RuntimeException("Cannot calculate t-test score for only one group");
        else {
            SampleMap::const_iterator s1Itr = itr->second.begin();
            SampleMap::const_iterator s2Itr = itr->second.begin();
            ++s2Itr;
            double m1 = mean(s1Itr->second);
            double m2 = mean(s2Itr->second);
            double s1 = variance(s1Itr->second);
            double s2 = variance(s2Itr->second);
            int n1 = (int) s1Itr->second.size();
            int n2 = (int) s2Itr->second.size();
            result[itr->first] = (m1 - m2) / 
                sqrt((n1 - 1) * s1 + (n2 - 1) * s2 / (n1 + n2 - 2) * (1 / n1 + 1 / n2));
        }
    }

    return result;
}


bool AnovaFeatureSelector::lowerScoreIsHigherRank()
{
    return false;
}


bool AnovaFeatureSelector::rankByAbsoluteScore()
{
    return true;
}


} // namespace blissart
