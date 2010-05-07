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


#ifndef __BLISSART_CORRELATION_FEATURE_SELECTOR_H__
#define __BLISSART_CORRELATION_FEATURE_SELECTOR_H__


#include <common.h>
#include <blissart/FeatureSelector.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Selects features by measuring correlation between their values and the
 * corresponding class labels.
 */
class LibFramework_API CorrelationFeatureSelector: public FeatureSelector
{
protected:
    /**
     * Implementation of FeatureSelector interface. Calculates the correlation
     * coefficient between the values of each feature and the class labels
     * assigned to the corresponding entities.
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
    // A vector that is labelled with double values.
    typedef std::pair<std::vector<double>, std::vector<double> > LabelledVec;

    // Calculates the correlation coefficient of vectors x and y.
    double correlation(const std::vector<double>& x, const std::vector<double>& y);
};


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_CORRELATION_FEATURE_SELECTOR_H__
