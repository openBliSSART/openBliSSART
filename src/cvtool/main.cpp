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


#include <Poco/Util/HelpFormatter.h>
#include <Poco/Exception.h>
#include <Poco/NumberFormatter.h>
#include <Poco/NumberParser.h>
#include <Poco/Util/RegExpValidator.h>

#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/BasicApplication.h>
#include <blissart/SVMModel.h>
#include <blissart/CorrelationFeatureSelector.h>
#include <blissart/AnovaFeatureSelector.h>
#include <blissart/CrossValidator.h>
#include <blissart/validators.h>
#include <blissart/exportDataSet.h>

#include <iostream>
#include <iomanip>
#include <map>
#include <ctime>
#include <cstdlib>
#include <algorithm>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using namespace Poco::Util;


// Avoid stupid Win32 "max" #define included from somewhere.
#if defined(_WIN32) || defined(_MSC_VER)
# ifdef max
#  undef max
# endif
#endif


class CrossValidationTool : public BasicApplication
{
public:
    CrossValidationTool() :
      BasicApplication(),
      _displayUsage(false),
      _responseID(0),
      _fold(10),
      _trainID(0),
      _shuffle(false),
      _upsample(false),
      _featureSelector(0),
      _maxFeatures(10),
      _probabilities(false),
      _isVerbose(false),
      _doDump(false),
      _dumpPrefix("fold")
    {
        addSubsystem(new DatabaseSubsystem());
    }


protected:
    typedef map<int, map<int, int> > ConfusionMatrix;


    void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);
        
        options.addOption(
            Option("help", "h",
                   "Displays usage information",
                   false));
        
        options.addOption(
            Option("response", "r",
                   "Validates the response with the given ID",
                   true, "<id>", true)
            .validator(new validators::RangeValidator<int>(1)));
        
        options.addOption(
            Option("fold", "f",
                   "Use n-fold SCV (0 for leave-one-out), default: 10", 
                   false, "<number>", true)
            .validator(new validators::RangeValidator<int>(0))
            .group("method"));

        options.addOption(
            Option("train", "t",
                   "Use data from the given response as training set",
                   false, "<id>", true)
            .validator(new validators::RangeValidator<int>(1))
            .group("method"));
        
        options.addOption(
            Option("upsample", "u",
                   "Upsamples the training set", false));

        options.addOption(
            Option("verbose", "v",
                   "Verbose output, "
                   "print all misclassified objects with their ID.",
                   false));

        options.addOption(
            Option("dump", "",
                   "Write training and test data for each fold to a text file "
                   "with the given prefix (default \"fold\").",
                   false, "<prefix>", false));

        options.addOption(
            Option("prob", "p",
                   "Estimate probabilities for classification. "
                   "Enables verbose output.",
                   false));

        options.addOption(
            Option("fs", "", 
                   "Perform feature selection. Possible algorithms include "
                   "\"anova\" and \"correlation\"",
                   false, "<algorithm>", true)
            .validator(new RegExpValidator("anova|correlation")));
        
        options.addOption(
            Option("max-features", "m",
                   "Maximum number of features to select. Default: 10",
                   false, "<number>", true)
            .validator(new validators::RangeValidator<int>(1)));
        
        options.addOption(
            Option("shuffle", "s",
                   "Shuffle data set to introduce some randomness. Only makes "
                   "sense if fold > 0.",
                   false));
    }


    void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);
        
        if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }
        else if (name == "response") {
            _responseID = Poco::NumberParser::parse(value.c_str());
        }
        else if (name == "fold") {
            _fold = Poco::NumberParser::parse(value.c_str());
        }
        else if (name == "train") {
            _trainID = Poco::NumberParser::parse(value.c_str());
        }
        else if (name == "verbose") {
            _isVerbose = true;
        }
        else if (name == "dump") {
            _doDump = true;
            if (!value.empty())
                _dumpPrefix = value;
        }
        else if (name == "prob") {
            _probabilities = true;
            _isVerbose = true;
        }
        else if (name == "fs") {
            if (value == "anova") {
                _featureSelector = new AnovaFeatureSelector;
            }
            else if (value == "correlation") {
                _featureSelector = new CorrelationFeatureSelector;
            }
        }
        else if (name == "max-features") {
            _maxFeatures = Poco::NumberParser::parse(value.c_str());
        }
        else if (name == "shuffle") {
            _shuffle = true;
        }
        else if (name == "upsample") {
            _upsample = true;
        }
    }


    void featureSelection(DataSet& dataSet, FeatureSet& featureSet)
    {
        featureSet.clear();
        FeatureSelector::FeatureScoreMap fs = 
            _featureSelector->filterDataSet(dataSet, _maxFeatures);
        cout << "Features selected:" << endl;
        for (FeatureSelector::FeatureScoreMap::const_iterator 
            itr = fs.begin(); itr != fs.end(); ++itr)
        {
            featureSet.add(itr->first);
            cout << itr->first.toString() << ": " << itr->second << endl;
        }
        cout << endl;
    }


    void validate()
    {
        DatabaseSubsystem& database = getSubsystem<DatabaseSubsystem>();
        ResponsePtr response = database.getResponse(_responseID);
        if (response.isNull()) {
            throw Poco::NotFoundException("No response with ID " +
                Poco::NumberFormatter::format(_responseID) + 
                " found in database");
        }

        DataSet dataSet;
        FeatureSet featureSet;
        string scalingMethod = config().
            getString("blissart.classification.scaling.method", "minmax");
        if (scalingMethod != "minmax" && scalingMethod != "musigma" 
            && scalingMethod != "none")
        {
            throw Poco::InvalidArgumentException(
                "Invalid scaling method specified: " + scalingMethod);
        }
        double lower = config().
            getDouble("blissart.classification.scaling.lower", -1.0);
        double upper = config().
            getDouble("blissart.classification.scaling.upper", 1.0);
        double mu = config().
            getDouble("blissart.classification.scaling.mu", 0.0);
        double sigma = config().
            getDouble("blissart.classification.scaling.sigma", 1.0);
        double accuracy = 0.0;

        // Fetch the label text for all label IDs occurring in the response
        // (used for output of confusion matrix, accuracies + upsample factors)
        vector<LabelPtr> labels = database.getLabelsForResponse(response);
        map<int, string> labelTextByID;
        for (vector<LabelPtr>::const_iterator itr = labels.begin();
            itr != labels.end(); ++itr)
        {
            labelTextByID[(*itr)->labelID] = (*itr)->text;
        }

        // Specific training response
        if (_trainID > 0) {
            ResponsePtr trainResponse = database.getResponse(_trainID);
            if (trainResponse.isNull()) {
                throw Poco::InvalidArgumentException(
                    "Invalid training response: " +
                    Poco::NumberFormatter::format(_trainID));
            }
            DataSet trainingSet; 
            // If feature selection is enabled, we ignore manual feature selection
            // in the config file and retrieve all available features at first
            if (_featureSelector) {
                trainingSet = database.getDataSet(trainResponse);
                featureSelection(trainingSet, featureSet);
            }
            // Otherwise, retrieve only the features specified in the config
            else {
                featureSet = FeatureSet::getStandardSet();
                trainingSet = database.getDataSet(trainResponse, featureSet);
            }
            // Retrieve test set with the same features as in the training set
            dataSet = database.getDataSet(response, featureSet);
            // Scale training + test set together
            if (scalingMethod != "none") {
                vector<DataSet> dataSets(2);
                dataSets[0] = trainingSet;
                dataSets[1] = dataSet;
                if (scalingMethod == "minmax")
                    linearScaleMinMax(dataSets, lower, upper);
                else if (scalingMethod == "musigma")
                    linearScaleMuSigma(dataSets, mu, sigma);
                trainingSet = dataSets[0];
                dataSet = dataSets[1];
            }

            // Upsample if desired.
            if (_upsample) {
                DataSet::size_type origSize = trainingSet.size();
                map<int, int> factors;
                upsample(trainingSet, factors);
                cout << "Upsampled training set from " << origSize << " to "
                     << trainingSet.size() << " samples. Factors:" << endl;
                for (map<int, int>::const_iterator fItr = factors.begin();
                    fItr != factors.end(); ++fItr)
                {
                    cout << setw(20) << labelTextByID[fItr->first] << ":" 
                         << fItr->second << endl;
                }
            }

            // Classify test set using SVM model from training set.
            SVMModel model(trainingSet);
            int nCorrect = model.classify(dataSet);
            accuracy = (double) nCorrect / (double) dataSet.size();
            cout << "Validated " << dataSet.size() 
                 << " samples with a training set of "
                 << trainingSet.size() << " samples" << endl;
        }

        // SCV
        else {
            if (_featureSelector) {
                dataSet = database.getDataSet(response);
                featureSelection(dataSet, featureSet);
            }
            else {
                featureSet = FeatureSet::getStandardSet();
                dataSet = database.getDataSet(response, featureSet);
            }
            
            if (scalingMethod != "none") {
                if (scalingMethod == "minmax")
                    linearScaleMinMax(dataSet, lower, upper);
                else if (scalingMethod == "musigma")
                    linearScaleMuSigma(dataSet, mu, sigma);
            }
            
            // Shuffle
            if (_shuffle) {
                srand((unsigned int) time(0));
                random_shuffle(dataSet.begin(), dataSet.end());
            }

            // Perform SCV.
            CrossValidator validator;
            if (_fold == 0) {
                accuracy = validator.loocv(dataSet, _probabilities);
            }
            else if (_fold > 0) {
                accuracy = validator.nFoldCrossValidation(dataSet, _fold, 
                    _probabilities);
            }

            cout << "Validated " << dataSet.size() << " samples with ";
            if (_fold == 0) {        
                cout << "LOOCV";
            }
            else {
                cout << _fold << "-fold cross validation"; 
            }
            cout << "." << endl;

            // Dump training/validation sets if desired
            if (_doDump && _trainID == 0) {
                cout << "Exporting training/validation sets... ";
                int nFolds = _fold > 0 ? _fold : (int) dataSet.size();
                string description = Poco::NumberFormatter::format(nFolds) +
                    "-fold cross-validation of '" + response->name + "'";
                for (int i = 0; i < nFolds; ++i) {
                    string foldNumber = Poco::NumberFormatter::format(i + 1);
                    exportDataSet(validator.trainingSet(i), 
                        _dumpPrefix + foldNumber + "_training.arff",
                        "Fold #" + foldNumber + ", training set", 
                        description + ", fold #" + foldNumber + ", training set");
                    exportDataSet(validator.validationSet(i), 
                        _dumpPrefix + foldNumber + "_validation.arff",
                        "Fold #" + foldNumber + ", validation set", 
                        description + ", fold #" + foldNumber + ", validation set");
                }
                cout << "done." << endl;
            }
        }

        // Build confusion matrix and output misclassifications, if desired
        ConfusionMatrix cm;
        map<int, int> classSize; // for recalls
        if (_isVerbose)
            cout << endl << "Misclassifications:" << endl;
        for (DataSet::const_iterator itr = dataSet.begin();
            itr != dataSet.end(); ++itr)
        {
            if (_isVerbose && itr->classLabel != itr->predictedClassLabel) {
                cout << "ID: " << itr->objectID << "; label = "
                     << labelTextByID[itr->classLabel] << "; predicted = "
                     << labelTextByID[itr->predictedClassLabel];
                if (_probabilities) {
                    cout << "; p = " << itr->predictionProbability;
                }
                cout << endl;
            }
            cm[itr->classLabel][itr->predictedClassLabel]++;
            classSize[itr->classLabel]++;
        }
        
        //
        // Output confusion matrix
        //

        cout << left << endl;
        
        // Get maximum length of labels. 
        // The output field width is the maximum length + 1.
        int width = 0;
        for (vector<LabelPtr>::const_iterator labelItr = labels.begin();
            labelItr != labels.end(); ++labelItr)
        {
            int length = (int) (*labelItr)->text.length();
            if (width < length)
                width = length;
        }
        ++width;
        
        // Header
        cout << "Confusion matrix" << endl << endl;
        cout << setw(max(width, 5)) << ""     << setw(width) << "predicted" << endl;
        cout << setw(max(width, 5)) << "real";
        for (vector<LabelPtr>::const_iterator labelItr = labels.begin();
            labelItr != labels.end(); ++labelItr)
        {
            cout << setw(width) << (*labelItr)->text;
        }
        cout << endl;

        // Rows
        for (vector<LabelPtr>::const_iterator rowItr = labels.begin();
            rowItr != labels.end(); ++rowItr)
        {
            cout << setw(max(width, 5)) << (*rowItr)->text;
            for (vector<LabelPtr>::const_iterator colItr = labels.begin();
                colItr != labels.end(); ++colItr)
            {
                cout << setw(width) 
                     << cm[(*rowItr)->labelID][(*colItr)->labelID];
            }
            cout << endl;
        }
        cout << endl;

        // Accuracy
        cout << "Accuracy = " << accuracy << endl;

        // Recalls
        cout << "Recalls:" << endl;
        map<int, double> recalls;
        double meanRecall = 0.0;
        for (vector<LabelPtr>::const_iterator labelItr = labels.begin();
            labelItr != labels.end(); ++labelItr)
        {
            int labelID = (*labelItr)->labelID;
            double recall = (double) cm[labelID][labelID] / (double) classSize[labelID];
            recalls[labelID] = recall;
            meanRecall += recall;
            cout << setw(width) << (*labelItr)->text << recall << endl;
        }
        meanRecall /= cm.size();
        cout << "Mean recall: " << meanRecall << endl;

    }


    int main(const vector<string> &args)
    {
        if (_displayUsage || args.size() != 0) {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() + " <options>\n");
            formatter.setHeader("CVTool, a cross validation tool");
            formatter.format(cout);
            return EXIT_USAGE;
        }
        
        validate();      

        if (_featureSelector) {
            delete _featureSelector;
            _featureSelector = 0;
        }

        return EXIT_OK;
    }


    bool              _displayUsage;
    int               _responseID;
    int               _fold;
    int               _trainID;
    bool              _shuffle;
    bool              _upsample;
    FeatureSelector*  _featureSelector;
    int               _maxFeatures;
    bool              _probabilities;
    bool              _isVerbose;
    bool              _doDump;
    string            _dumpPrefix;
};


POCO_APP_MAIN(CrossValidationTool);

