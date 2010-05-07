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


#ifndef __BLISSART_FEATURE_SELECTOR_H__
#define __BLISSART_FEATURE_SELECTOR_H__


#include <common.h>
#include <map>
#include <utility>
#include <vector>
#include <blissart/FeatureDescriptor.h>
#include <blissart/FeatureSet.h>
#include <blissart/DataSet.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Abstract base class for classes that provides feature selection based on 
 * a feature metric.
 */
class LibFramework_API FeatureSelector
{
public:
    /*
     * Assigns a score to each feature.
     */
    typedef std::map<FeatureDescriptor, double>  FeatureScoreMap;
    
    /**
     * Rates the features in the given data set and removes all but the best n.
     */
    FeatureScoreMap filterDataSet(DataSet& dataSet, int n);
    
    virtual ~FeatureSelector() {}

protected:
    // Since this is an abstract base class, we forbid the default constructor.
    FeatureSelector() {}

    // Assigns a rank to each feature.
    typedef std::map<FeatureDescriptor, int>     FeatureRankMap;
    
    // Representation of feature scores as vector, used for sorting,
    // since maps cannot be sorted.
    typedef std::pair<FeatureDescriptor, double> FeatureScore;
    typedef std::vector<FeatureScore>            FeatureScoreVec;

    /**
     * Rates the features in the given DataSet. Must be implemented by
     * subclasses.
     */
    virtual FeatureScoreMap rateFeatures(const DataSet& dataSet) = 0;
    
    /**
     * Returns true if a lower score means better score, false otherwise.
     * Must be implemented by subclasses.
     */ 
    virtual bool lowerScoreIsHigherRank() = 0;

    /**
     * Returns true if ranks depend on the absolute value of the feature score.
     */
    virtual bool rankByAbsoluteScore() = 0;

    /**
     * Helper function used for sorting.
     */
    static bool compareByScore(const FeatureScore& f1, const FeatureScore& f2);

    /**
     * Helper function used for sorting.
     */
    static bool compareByScoreDesc(const FeatureScore& f1, const FeatureScore& f2);

    // XXX: We have these functions in LibFeature as well, but for linalg::Vector.
    // However, we need to use std::vector as input since the sample size is not
    // known a priori.
    double mean(const std::vector<double>& s);
    double variance(const std::vector<double>& s);
};


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_FEATURE_SELECTOR_H__
