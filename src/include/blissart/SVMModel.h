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


#ifndef __BLISSART_SVMMODEL_H__
#define __BLISSART_SVMMODEL_H__


#include <common.h>
#include <map>
#include <utility>
#include <string>
#include <vector>
#include <blissart/DataSet.h>
#include <blissart/FeatureDescriptor.h>
#include <blissart/FeatureSet.h>

#include <libsvm/svm.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Creates SVM models from DataSets using LibSVM and classifies DataSets.
 */
class LibFramework_API SVMModel
{
public:
    /**
     * Creates a SVMModel that models the given DataSet.
     * If probabilitySupport is true, a model is created that supports
     * estimation of classification probabilities, which will take more time.
     */
    SVMModel(const DataSet& dataSet, bool probabilitySupport = false);
    
    /**
     * Destroys the SVMModel and frees all memory used by it.
     */
    virtual ~SVMModel();

    /**
     * Sets the class labels of the given DataSet according to the SVM.
     * Returns the number of correct predictions.
     */
    int classify(DataSet& dataSet, bool estimateProbabilities = false);

private:
    // Unimplemented
    SVMModel();
    SVMModel(const SVMModel&);

    bool             _probabilitySupport;
    bool             _hasBias;

    FeatureSet       _featureSet;
    
    // The data used to train the SVM model. It is referenced by the svm_model
    // structure below and has therefore to be kept until the model is 
    // destroyed.
    svm_problem      _svmProblem;
    svm_node*        _svmNodes;
    
    svm_model*       _svmModel;
};


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_SVMMODEL_H__
