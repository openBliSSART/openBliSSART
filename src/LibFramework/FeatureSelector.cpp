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


#include <blissart/FeatureSelector.h>
#include <cmath>


using namespace std;


namespace blissart {


FeatureSelector::FeatureScoreMap 
FeatureSelector::filterDataSet(DataSet& dataSet, int n)
{
    if (n < 1)
        throw Poco::InvalidArgumentException("Must select at least one feature");

    FeatureScoreMap result;

    // Obtain scores and convert them to a vector, which can be sorted 
    // (a map cannot).
    FeatureScoreMap scores = rateFeatures(dataSet);
    FeatureScoreVec scoresVec;
    for (FeatureScoreMap::const_iterator itr = scores.begin(); 
        itr != scores.end(); ++itr)
    {
        if (rankByAbsoluteScore())
            scoresVec.push_back(FeatureScore(itr->first, fabs(itr->second)));
        else
            scoresVec.push_back(FeatureScore(itr->first, itr->second));
    }

    if (lowerScoreIsHigherRank())
        sort(scoresVec.begin(), scoresVec.end(), &compareByScore);
    else
        sort(scoresVec.begin(), scoresVec.end(), &compareByScoreDesc);
    
    // Iterate over the scores vector and assign ranks to each feature,
    // according to its position in the vector (lower rank means better).
    int rank = 0;
    FeatureRankMap featureRanks;
    for (FeatureScoreVec::const_iterator itr = scoresVec.begin();
        itr != scoresVec.end(); ++itr)
    {
        featureRanks[itr->first] = rank++;
    }

    // Erase all features that rank too bad from the DataSet.
    for (DataSet::iterator itr = dataSet.begin(); itr != dataSet.end(); ++itr)
    {
        DataPoint::ComponentMap::iterator compItr = itr->components.begin();
        while (compItr != itr->components.end()) {
            // If the feature ranks too bad, erase it and set the iterator
            // to the next available item.
            if (featureRanks[compItr->first] >= n) {
                itr->components.erase(compItr++);
            }
            // Otherwise, increment the iterator and insert the feature
            // into the result.
            else {
                if (result.find(compItr->first) == result.end())
                    result[compItr->first] = scores[compItr->first];
                ++compItr;
            }
        }
    }

    return result;
}


bool FeatureSelector::compareByScore(const FeatureScore& f1, const FeatureScore& f2)
{
    return f1.second < f2.second;
}

        
bool FeatureSelector::compareByScoreDesc(const FeatureScore& f1, const FeatureScore& f2)
{
    return f1.second > f2.second;
}

        
double FeatureSelector::mean(const vector<double>& s)
{
    double m = 0.0;
    for (vector<double>::const_iterator itr = s.begin(); itr != s.end(); ++itr) {
        m += *itr;
    }
    return m / (int) s.size();
}


double FeatureSelector::variance(const vector<double>& s)
{
    double m = mean(s);
    double v = 0.0;
    for (vector<double>::const_iterator itr = s.begin(); itr != s.end(); ++itr) {
        double dev = *itr - m;
        v += dev * dev;
    }
    return v / ((int) s.size() - 1);
}


} // namespace blissart
