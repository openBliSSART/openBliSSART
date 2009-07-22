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


#ifndef __BLISSART_CROSSVALIDATOR_H__
#define __BLISSART_CROSSVALIDATOR_H__


#include <common.h>
#include <blissart/DataSet.h>
#include <vector>


namespace blissart {


/** 
 * Performs stratified cross-validation on a DataSet.
 */
class LibFramework_API CrossValidator
{
public:
    CrossValidator();
    virtual ~CrossValidator();

    /**
     * Validates a DataSet using n-fold cross validation.
     */
    double nFoldCrossValidation(DataSet& dataSet, int fold,
        bool estimateProbabilities = false);

    /**
     * Validates a DataSet using leave-one-out cross validation.
     */
    double loocv(DataSet& dataSet, bool estimateProbabilities = false);

    /**
     * Returns the DataSet that was used to train the model in the fold with
     * the given index.
     */
    const DataSet& trainingSet(unsigned int index) const;

    /**
     * Returns the DataSet that was classified in the fold with
     * the given index.
     */
    const DataSet& validationSet(unsigned int index) const;

private:
    // Helper function to sort a DataSet by class label, needed for 
    // stratified cross-validation.
    static bool compareByClassLabel(const DataPoint& p1, const DataPoint& p2);

    std::vector<DataSet> _trainingSets;
    std::vector<DataSet> _validationSets;
};


} // namespace blissart


#endif // __BLISSART_CROSSVALIDATOR_H__
