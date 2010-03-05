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


#include <blissart/SVMModel.h>

#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/Util/Application.h>

#include <algorithm>


using namespace std;
using Poco::Util::LayeredConfiguration;


namespace blissart {


SVMModel::SVMModel(const DataSet& dataSet, bool probabilitySupport) :
    _probabilitySupport(probabilitySupport),
    _svmModel(0)
{
    if (dataSet.empty())
        throw Poco::InvalidArgumentException("Cannot train SVM on empty data set");

    //
    // Setup parameters for SVM training
    // Some parameters can be specified via the Application's configuration.
    //
    svm_parameter param;
    LayeredConfiguration& appConfig = Poco::Util::Application::instance().config();
    param.svm_type = C_SVC;

    string kernel = appConfig.getString("blissart.classification.svm.kernel", "linear");
    if (kernel == "linear")
        param.kernel_type = LINEAR;
    else if (kernel == "rbf")
        param.kernel_type = RBF;
    else if (kernel == "poly")
        param.kernel_type = POLY;
    else if (kernel == "sigmoid")
        param.kernel_type = SIGMOID;
    else
        throw Poco::InvalidArgumentException("Invalid SVM kernel type: " + kernel);

    param.degree = appConfig.getInt("blissart.classification.svm.degree", 3);
    if (param.degree < 1)
        throw Poco::InvalidArgumentException("Invalid polynomial degree");

    param.coef0 = appConfig.getDouble("blissart.classification.svm.coef0", 0.0);

    param.eps = appConfig.getDouble("blissart.classification.svm.epsilon", 1e-3);

    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = probabilitySupport ? 1 : 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    _hasBias = appConfig.getBool("blissart.classification.addBias", false);

    //
    // Convert the DataSet into a svm_problem structure
    //
    
    // Iterate over the data set to determine the space that we must allocate
    // for the nodes
    size_t nNodes = 0;
    for (DataSet::const_iterator dataSetItr = dataSet.begin();
         dataSetItr != dataSet.end(); ++dataSetItr)
    {
        DataPoint::ComponentMap::size_type nComponents = 
            dataSetItr->components.size();
        // Every data point is terminated by a "-1" component in libsvm's 
        // input format, so we additionally need space for that.
        nNodes += nComponents + 1;
        // If desired, reserve space for the bias.
        if (_hasBias)
            ++nNodes;
    }

    // Array of nodes (index-value pairs)
    _svmNodes = new svm_node[nNodes];

    // Length of the problem
    _svmProblem.l = (int) dataSet.size();
    
    // Class "labels" (as double values)
    _svmProblem.y = new double[_svmProblem.l];
    
    // Pointers to instances within the node array
    _svmProblem.x = new svm_node*[_svmProblem.l];
    
    // Instance index
    int i = 0;

    // Offset in node array
    int j = 0;
    
    for (DataSet::const_iterator dataSetItr = dataSet.begin();
         dataSetItr != dataSet.end(); ++dataSetItr)
    {
        _svmProblem.x[i] = &(_svmNodes[j]);
        _svmProblem.y[i] = (double) dataSetItr->classLabel;
    
        // Insert bias, if desired.
        if (_hasBias) {
            _svmNodes[j].index = 0;
            _svmNodes[j].value = 1.0;
            ++j;
        }
        
        for (DataPoint::ComponentMap::const_iterator compItr = 
             dataSetItr->components.begin();
             compItr != dataSetItr->components.end(); ++compItr)
        {
            if (!_featureSet.has(compItr->first)) {
                _featureSet.add(compItr->first);
            }
            _svmNodes[j].index = _featureSet.indexOf(compItr->first);

            // If we add a bias, it got index 0, so we shift all other 
            // indices by 1
            if (_hasBias)
                ++_svmNodes[j].index;

            _svmNodes[j].value = compItr->second;
            ++j;
        }

        // Insert a "-1" element to terminate the data point
        _svmNodes[j].index = -1;
        _svmNodes[j].value = 0;
        ++j;

        ++i;
    }

    //
    // Train the SVM model
    //
    if (_hasBias) {
        param.gamma = 1.0 / (double) _featureSet.size();
    }
    else {
        param.gamma = 1.0 / (double) (_featureSet.size() - 1);
    }
    _svmModel = svm_train(&_svmProblem, &param);
}


SVMModel::~SVMModel()
{
    if (_svmModel) {
        svm_destroy_model(_svmModel);
        _svmModel = 0;
    }
    delete[] _svmProblem.x;
    delete[] _svmProblem.y;
    delete[] _svmNodes;
}


int SVMModel::classify(DataSet &dataSet, bool estimateProbabilities)
{
    if (estimateProbabilities && !_probabilitySupport) {
        throw Poco::IllegalStateException("SVM model was not build with probability support");
    }

    int i, nCorrect = 0;

    for (DataSet::iterator dataSetItr = dataSet.begin();
         dataSetItr != dataSet.end(); ++dataSetItr)
    {
        // Create array of svm_nodes for the current data point
        size_t nNodes = dataSetItr->components.size() + 1;
        if (_hasBias)
            ++nNodes;
        svm_node* nodes = new svm_node[nNodes];

        i = 0;
        if (_hasBias) {
            nodes[i].index = 0;
            nodes[i].value = 1;
            ++i;
        }

        // Copy the data from the DataSet
        for (DataPoint::ComponentMap::const_iterator compItr = 
            dataSetItr->components.begin();
            compItr != dataSetItr->components.end(); ++compItr)
        {
            if (!_featureSet.has(compItr->first)) {
                throw Poco::NotFoundException("SVM model was not built with feature " + 
                    compItr->first.toString() + " for type " + 
                    DataDescriptor::strForType(compItr->first.dataType));
            }
            nodes[i].index = _featureSet.indexOf(compItr->first);

            // We have to respect the bias flag here as well.
            if (_hasBias)
                ++nodes[i].index;
            
            nodes[i].value = compItr->second;
            ++i;
        }
        
        nodes[i].index = -1;
        nodes[i].value = 0;

        double result;
        
        if (estimateProbabilities && _probabilitySupport) {
            // Get the probabilities for each label and assign the label with the
            // highest probability
            int nClasses = svm_get_nr_class(_svmModel);
            double* probabilities = new double[nClasses];
            result = svm_predict_probability(_svmModel, nodes, probabilities);
            int maxIndex = 0;
            for (int cl = 1; cl < nClasses; ++cl) {
                if (probabilities[maxIndex] < probabilities[cl]) {
                    maxIndex = cl;
                }
            }
            dataSetItr->predictionProbability = probabilities[maxIndex];
        }

        else {
            result = svm_predict(_svmModel, nodes);
        }

        dataSetItr->predictedClassLabel = (int) result;
        if (dataSetItr->predictedClassLabel == dataSetItr->classLabel)
            ++nCorrect;

        delete[] nodes;
    }

    return nCorrect;
}


} // namespace blissart
