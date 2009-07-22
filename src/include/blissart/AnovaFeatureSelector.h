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


#ifndef __BLISSART_ANOVA_FEATURE_SELECTOR_H__
#define __BLISSART_ANOVA_FEATURE_SELECTOR_H__


#include <common.h>
#include <blissart/FeatureSelector.h>


namespace blissart {


/**
 * Selects features by performing analysis of variance (ANOVA).
 */
class LibFramework_API AnovaFeatureSelector: public FeatureSelector
{
protected:
    /**
     * Implementation of FeatureSelector interface.
     * Calculates the t-test score if the DataSet contains 2 groups.
     * Throws a Poco::NotImplementedException otherwise.
     */
    FeatureScoreMap rateFeatures(const DataSet& dataSet);
    
    /**
     * Implementation of FeatureSelector interface. Always returns false.
     */
    bool lowerScoreIsHigherRank();

    /**
     * Implementation of FeatureSelector interface. Always returns true.
     */
    bool rankByAbsoluteScore();

private:
    typedef std::vector<double> Sample;

    // Maps a class label to a sample.
    typedef std::map<int, Sample> SampleMap;

};


} // namespace blissart


#endif // __BLISSART_ANOVA_FEATURE_SELECTOR_H__
