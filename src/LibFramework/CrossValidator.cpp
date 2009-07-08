//
// $Id: CrossValidator.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <blissart/CrossValidator.h>
#include <blissart/SVMModel.h>


using std::vector;


namespace blissart {


CrossValidator::CrossValidator()
{
}


CrossValidator::~CrossValidator()
{
}


double CrossValidator::nFoldCrossValidation(DataSet& dataSet, int fold,
                                            bool estimateProbabilities)
{
    int nCorrect = 0;
    _trainingSets.clear();
    _validationSets.clear();

    sort(dataSet.begin(), dataSet.end(), CrossValidator::compareByClassLabel);

    // Create training/validation pairs;
    // for each pair, a SVM model is built from the training set
    // and the validation set is classified
    for (int setIndex = 0; setIndex < fold; ++setIndex)
    {
        DataSet trainingSet;
        DataSet validationSet;
        
        // For each item in the validation set, store the positions of 
        // the corresponding item in the complete data set
        vector<int> validationItemPos;

        int pos = 0;
        for (DataSet::const_iterator itr = dataSet.begin();
            itr != dataSet.end(); ++itr, ++pos)
        {
            if (pos % fold == setIndex) {
                validationSet.push_back(*itr);
                validationItemPos.push_back(pos);
            }
            else {
                trainingSet.push_back(*itr);
            }
        }

        SVMModel model(trainingSet, estimateProbabilities);
        nCorrect += model.classify(validationSet, estimateProbabilities);
        
        // Copy the predictions to the original data set
        int itemIndex = 0;
        for (DataSet::const_iterator itr = validationSet.begin();
            itr != validationSet.end(); ++itr, ++itemIndex)
        {
            dataSet[validationItemPos[itemIndex]].predictedClassLabel =
                itr->predictedClassLabel;
            if (estimateProbabilities) {
                dataSet[validationItemPos[itemIndex]].predictionProbability =
                    itr->predictionProbability;
            }
        }

        // Save training and validation set
        _trainingSets.push_back(trainingSet);
        _validationSets.push_back(validationSet);
    }

    return (double) nCorrect / dataSet.size();
}


double CrossValidator::loocv(DataSet &dataSet, bool estimateProbabilities)
{
    int nCorrect = 0;

    _trainingSets.clear();
    _validationSets.clear();

    for (DataSet::size_type i = 0; i < dataSet.size(); ++i) {
        DataSet trainingSet = dataSet;
        DataSet validationSet(1, dataSet[i]);
        trainingSet.erase(trainingSet.begin() + i);

        _trainingSets.push_back(trainingSet);
        _validationSets.push_back(validationSet);

        SVMModel model(trainingSet);
        nCorrect += model.classify(validationSet, false);

        dataSet[i].predictedClassLabel = validationSet[0].predictedClassLabel;
    }

    return (double) nCorrect / dataSet.size();
}


const DataSet& CrossValidator::trainingSet(unsigned int index) const
{
    if (index >= _trainingSets.size())
        throw Poco::InvalidArgumentException("index out of range");
    return _trainingSets.at(index);
}


const DataSet& CrossValidator::validationSet(unsigned int index) const
{
    if (index >= _validationSets.size())
        throw Poco::InvalidArgumentException("index out of range");
    return _validationSets.at(index);
}


bool CrossValidator::compareByClassLabel(const DataPoint &p1, 
                                         const DataPoint &p2)
{
    return p1.classLabel < p2.classLabel;
}


} // namespace blissart
