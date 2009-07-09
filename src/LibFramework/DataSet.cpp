//
// $Id: DataSet.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <blissart/DataSet.h>
#include <Poco/Exception.h>
#include <algorithm>
#include <utility>
#include <cmath>


using std::vector;
using std::pair;
using std::map;


namespace blissart {


DataPoint::DataPoint() :
    objectID(0),
    classLabel(0),
    predictedClassLabel(0),
    predictionProbability(0.0)
{
}


typedef map<FeatureDescriptor, vector<double> > FeatureValueMap;


void getValuesByFeature(const DataSet& dataSet, FeatureValueMap& featureValues)
{
    for (DataSet::const_iterator dItr = dataSet.begin(); 
        dItr != dataSet.end(); ++dItr)
    {
        for (DataPoint::ComponentMap::const_iterator cItr = dItr->components.begin();
            cItr != dItr->components.end(); ++cItr)
        {
            featureValues[cItr->first].push_back(cItr->second);
        }
    }
}


void linearScaleMinMax(vector<DataSet>& dataSets, double lower, double upper)
{
    if (lower >= upper)
        throw Poco::InvalidArgumentException("Lower bound must be strictly smaller than upper bound");
    
    FeatureValueMap featureValues;
    for (vector<DataSet>::const_iterator itr = dataSets.begin();
        itr != dataSets.end(); ++itr) 
    {
        getValuesByFeature(*itr, featureValues);
    }

    map<FeatureDescriptor, double> featureMin;
    map<FeatureDescriptor, double> featureMax;

    double targetDist = upper - lower;
    
    for (FeatureValueMap::const_iterator itr = featureValues.begin();
        itr != featureValues.end(); ++itr)
    {
        double min = *min_element(itr->second.begin(), itr->second.end());
        double max = *max_element(itr->second.begin(), itr->second.end());
        featureMin[itr->first] = min;
        featureMax[itr->first] = max;
        if (max == min) {
            throw Poco::InvalidArgumentException(
                "Minimal element equal to maximal element for feature " +
                itr->first.toString()
                );
        }
    }

    for (vector<DataSet>::iterator vItr = dataSets.begin(); 
        vItr != dataSets.end(); ++vItr)
    {
        for (DataSet::iterator dItr = vItr->begin(); 
            dItr != vItr->end(); ++dItr)
        {
            for (DataPoint::ComponentMap::iterator cItr = dItr->components.begin();
                cItr != dItr->components.end(); ++cItr)
            {
                cItr->second = lower + targetDist / 
                    (featureMax[cItr->first] - featureMin[cItr->first]) *
                    (cItr->second - featureMin[cItr->first]);
            }
        }
    }
}


void linearScaleMinMax(DataSet& dataSet, double lower, double upper)
{
    // XXX: This involves copying, but it's probably better than to mess around with
    // pointers, or having duplicate code ...
    vector<DataSet> dv(1, dataSet);
    linearScaleMinMax(dv, lower, upper);
    dataSet = dv[0];
}


void linearScaleMuSigma(vector<DataSet>& dataSets, double mu, double sigma)
{
    if (sigma <= 0)
        throw Poco::InvalidArgumentException("Invalid sigma");
    
    FeatureValueMap featureValues;
    for (vector<DataSet>::const_iterator itr = dataSets.begin();
        itr != dataSets.end(); ++itr) 
    {
        getValuesByFeature(*itr, featureValues);
    }

    map<FeatureDescriptor, double> featureMu;
    map<FeatureDescriptor, double> featureSigma;

    for (FeatureValueMap::const_iterator itr = featureValues.begin();
        itr != featureValues.end(); ++itr)
    {
        double sum   = 0.0;
        double sumSq = 0.0;
        for (vector<double>::const_iterator vItr = itr->second.begin();
            vItr != itr->second.end(); ++vItr)
        {
            sum   += *vItr;
            sumSq += *vItr * *vItr;
        }
        double factor = 1 / (double) itr->second.size();
        double muF = factor * sum;
        featureMu[itr->first] = muF;
        featureSigma[itr->first] = sqrt(factor * sumSq - muF * muF);
    }

    for (vector<DataSet>::iterator vItr = dataSets.begin(); 
        vItr != dataSets.end(); ++vItr)
    {
        for (DataSet::iterator dItr = vItr->begin(); 
            dItr != vItr->end(); ++dItr)
        {
            for (DataPoint::ComponentMap::iterator cItr = dItr->components.begin();
                cItr != dItr->components.end(); ++cItr)
            {
                double sigmaCorr = sigma / featureSigma[cItr->first];
                double muCorr = mu - featureMu[cItr->first] * sigmaCorr;
                cItr->second = cItr->second * sigmaCorr + muCorr;
            }
        }
    }
}


void linearScaleMuSigma(DataSet& dataSet, double mu, double sigma)
{
    vector<DataSet> dv(1, dataSet);
    linearScaleMuSigma(dv, mu, sigma);
    dataSet = dv[0];
}


inline double sigmoid(double x, double a, double b)
{
    return a / (1.0 + exp(-x)) + b;
}


void sigmoidSmooth(DataSet& dataSet, double lower, double upper)
{
    if (lower >= upper)
        throw Poco::InvalidArgumentException("Lower bound must be strictly smaller than upper bound");
    
    double dist = lower - upper;
    for (DataSet::iterator dItr = dataSet.begin(); dItr != dataSet.end(); ++dItr)
    {
        for (DataPoint::ComponentMap::iterator cItr = dItr->components.begin();
            cItr != dItr->components.end(); ++cItr)
        {
            cItr->second = sigmoid(cItr->second, dist, lower);
        }
    }
}


} // namespace blissart
