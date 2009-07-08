//
// $Id: DataSet.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_DATASET_H__
#define __BLISSART_DATASET_H__


#include <vector>
#include <map>
#include <string>
#include <blissart/FeatureDescriptor.h>


namespace blissart {


/**
 * A labelled data point in the feature space, corresponding to a 
 * ClassificationObject. It is stored as a sparse vector.
 * Each dimension of the feature space is described by a FeatureDescriptor.
 */
class LibFramework_API DataPoint
{
public:
    DataPoint();

    typedef std::map<FeatureDescriptor, double> ComponentMap;

    int           objectID;
    ComponentMap  components;
    int           classLabel;
    int           predictedClassLabel;
    double        predictionProbability;
};


/**
 * A vector of DataPoints.
 */
typedef std::vector<DataPoint> DataSet;


/**
 * Scales the given DataSet such that all feature values are in the given range,
 * using a linear function for each feature.
 * @param  lower   the value of the minimal elements in the target set
 * @param  upper   the value of the maximal elements in the target set
 */
void LibFramework_API 
linearScaleMinMax(DataSet& dataSet, double lower = -1.0, double upper = 1.0);


/**
 * Scales the given DataSets such that all feature values are in the given range,
 * using a linear function for each feature.
 * @param  lower   the value of the minimal elements in the target set
 * @param  upper   the value of the maximal elements in the target set
 */
void LibFramework_API 
linearScaleMinMax(std::vector<DataSet>& dataSets, double lower = -1.0, 
                  double upper = 1.0);


/**
 * Scales the given DataSet such that the values of each feature have the given
 * mean and standard deviation, using linear functions.
 * @param  mu      the desired mean
 * @param  upper   the desired standard deviation
 */
void LibFramework_API
linearScaleMuSigma(DataSet& dataSet, double mu = 0.0, double sigma = 1.0);


/**
 * Scales the given DataSets such that the values of each feature have the given
 * mean and standard deviation, using linear functions.
 * @param  mu      the desired mean
 * @param  upper   the desired standard deviation
 */
void LibFramework_API
linearScaleMuSigma(std::vector<DataSet>& dataSets, double mu = 0.0,
                   double sigma = 1.0);


/**
 * Smoothes the given DataSet using a sigmoid function.
 * @param   lower   the ordinate of the lower asymptote
 * @param   upper   the ordinate of the upper asymptote
 */
void LibFramework_API 
sigmoidSmooth(DataSet& dataSet, double lower = -1.0, double upper = 1.0);


} // namespace blissart


#endif // __BLISSART_DATASET_H__
