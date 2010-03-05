//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
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


#include "SVMModelTest.h"

#include <blissart/SVMModel.h>
#include <blissart/DataSet.h>
#include <blissart/DataDescriptor.h>
#include <blissart/CrossValidator.h>

#include <Poco/Util/Application.h>

#include <iostream>
#include <ctime>
#include <cstdlib>


using namespace blissart;
using namespace std;


namespace Testing {


void SVMModelTest::addPoint(DataSet& dataSet, double x, double y, int classLabel)
{
    DataPoint point;

    FeatureDescriptor fX("stddev", DataDescriptor::Spectrum);
    FeatureDescriptor fY("stddev", DataDescriptor::Gains);

    point.classLabel = classLabel;
    point.components[fX] = x;
    point.components[fY] = y;

    dataSet.push_back(point);
}


bool SVMModelTest::performTest()
{
    Poco::Util::Application::instance().config().setString(
        "blissart.classification.svm.kernel", "rbf");
    DataSet dataSet;

    addPoint(dataSet, -1, 2, 1);
    addPoint(dataSet, -2, 1, 1);
    addPoint(dataSet, -0.5, 4, 1);
    addPoint(dataSet, 0.5, 2.5, 1);
    addPoint(dataSet, 0.7, 2, 1);
    addPoint(dataSet, 0.5, 0, 1);

    addPoint(dataSet, -1, -2, 2);
    addPoint(dataSet, 2.5, 1, 2);
    addPoint(dataSet, 1, -0.5, 2);
    addPoint(dataSet, 3, -2, 2);
    addPoint(dataSet, 3, 3.5, 2);

    addPoint(dataSet, 10, 20, 3);
    addPoint(dataSet, 15, 19, 3);
    addPoint(dataSet, 16, 17, 3);
    addPoint(dataSet, 21, 21, 3);
    addPoint(dataSet, 14, 17, 3);

    CrossValidator validator;
    double accuracy = validator.nFoldCrossValidation(dataSet, 3);
    cout << "3-fold cross validation " << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << endl;

    accuracy = validator.loocv(dataSet);
    cout << "Leave-one-out cross validation " << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << endl;

    DataSet testSet;
    addPoint(testSet, -2, -2, 0);
    addPoint(testSet, -1, 1.5, 0);
    addPoint(testSet, 15, 16, 0);

    cout << "Classification (without probability estimates)" << endl;
    SVMModel model(dataSet, false);
    model.classify(testSet, false);
    cout << "Class label for test point #1: " << testSet[0].predictedClassLabel << endl;
    cout << "Class label for test point #2: " << testSet[1].predictedClassLabel << endl;
    cout << "Class label for test point #3: " << testSet[2].predictedClassLabel << endl;

    if (testSet[0].predictedClassLabel != 2 || testSet[1].predictedClassLabel != 1)
        return false;

    cout << "Classification (with probability estimates)" << endl;
    // Calculation of probability estimates is a randomized process.
    // We therefore have to initialize the random generator with a fixed value
    // to make it deterministic.
    srand(0);
    SVMModel modelWithProb(dataSet, true);
    modelWithProb.classify(testSet, true);
    cout << "Class label for test point #1: " << testSet[0].predictedClassLabel 
         << " (probability: " << testSet[0].predictionProbability << ")" << endl;
    cout << "Class label for test point #2: " << testSet[1].predictedClassLabel
         << " (probability: " << testSet[1].predictionProbability << ")" << endl;
    cout << "Class label for test point #3: " << testSet[2].predictedClassLabel
         << " (probability: " << testSet[2].predictionProbability << ")" << endl;

    // The probability values are tested against those obtained with libsvm's 
    // "svm-predict" implementation.
    if (testSet[0].predictedClassLabel != 3 || 
        testSet[1].predictedClassLabel != 2 ||
        testSet[2].predictedClassLabel != 2 ||
        !epsilonCheck(testSet[0].predictionProbability, 0.451, 1e-3) || 
        !epsilonCheck(testSet[1].predictionProbability, 0.375, 1e-3) ||
        !epsilonCheck(testSet[2].predictionProbability, 0.400, 1e-3))
    {
        return false;
    }

    cout << "Classification with bias" << endl;
    Poco::Util::Application::instance().config().setBool(
        "blissart.classification.addBias", true);
    srand(0);
    SVMModel modelWithBias(dataSet, true);
    modelWithBias.classify(testSet, true);
    cout << "Class label for test point #1: " << testSet[0].predictedClassLabel 
         << " (probability: " << testSet[0].predictionProbability << ")" << endl;
    cout << "Class label for test point #2: " << testSet[1].predictedClassLabel
         << " (probability: " << testSet[1].predictionProbability << ")" << endl;
    cout << "Class label for test point #3: " << testSet[2].predictedClassLabel
         << " (probability: " << testSet[2].predictionProbability << ")" << endl;

    if (testSet[0].predictedClassLabel != 3 || 
        testSet[1].predictedClassLabel != 1 ||
        testSet[2].predictedClassLabel != 2 ||
        !epsilonCheck(testSet[0].predictionProbability, 0.567, 1e-3) || 
        !epsilonCheck(testSet[1].predictionProbability, 0.625, 1e-3) ||
        !epsilonCheck(testSet[2].predictionProbability, 0.551, 1e-3))
    {
        return false;
    }

    return true;
}


} // namespace Testing
