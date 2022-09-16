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


#include <blissart/ClassificationTask.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/Response.h>
#include <blissart/SVMModel.h>
#include <blissart/FeatureSet.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/generators/generators.h>
#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>

#include <Poco/Util/LayeredConfiguration.h>

#include <memory>


using namespace blissart::linalg;
using namespace blissart::audio;
using namespace std;
using Poco::Util::LayeredConfiguration;


namespace blissart {


ClassificationTask::ClassificationTask(int responseID,
                                       const linalg::Matrix& phaseMatrix,
                                       const vector<Matrix*>& componentSpectrograms,
                                       const linalg::Matrix& gainsMatrix,
                                       int sampleRate, int windowSize,
                                       double overlap,
                                       const string& fileName) :
    BasicTask("ClassificationTask"),
    _responseID(responseID),
    _phaseMatrix(&phaseMatrix),
    _componentSpectrograms(componentSpectrograms),
    _gainsMatrix(&gainsMatrix),
    _sampleRate(sampleRate),
    _windowSize(windowSize),
    _overlap(overlap),
    _fileName(fileName),
    _dataSet(0)
{
    _extractor.setSampleFrequency(_sampleRate);
    // We have the window size in ms!
    _extractor.setGainsFrequency(1000.0 / (_windowSize * (1.0 - _overlap)));
}


ClassificationTask::ClassificationTask(int responseID,
                                       const SeparationTaskPtr& sepTask) :
    BasicTask("ClassificationTask"),
    _responseID(responseID),
    _sepTask(sepTask),
    _phaseMatrix(0),
    _componentSpectrograms(sepTask->nrOfComponents()),
    _gainsMatrix(0),
    _windowSize(sepTask->windowSize()),
    _overlap(sepTask->overlap()),
    _dataSet(0)
{
    // The sample rate is not known until SeparationTask has been started
    // and read the audio file! Hence it will be set in runTask().

    // We have the window size in ms!
    _extractor.setGainsFrequency(1000.0 / (_windowSize * (1.0 - _overlap)));
}


ClassificationTask::~ClassificationTask()
{
    for (map<int, Matrix*>::const_iterator it = _spectraMap.begin();
        it != _spectraMap.end(); ++it)
    {
        delete it->second;
     }
    _spectraMap.clear();
}


void ClassificationTask::presetClassLabel(int componentIndex, int classLabel)
{
    _presetClassLabels[componentIndex] = classLabel;
}


void ClassificationTask::runTask()
{
    incMaxProgress(1.0f);

    // If this ClassificationTask has been constructed for a corresponding
    // SeparationTask, all neccessary data have to be fetched now.
    if (!_sepTask.isNull()) {
        _phaseMatrix    = &_sepTask->phaseMatrix();
        _gainsMatrix    = &_sepTask->gainsMatrix();
        _sampleRate     = _sepTask->sampleRate();
        _fileName       = _sepTask->fileName();
        _extractor.setSampleFrequency(_sampleRate);
        unsigned int compIndex = 0;
        for (vector<Matrix*>::iterator it = _componentSpectrograms.begin();
            it != _componentSpectrograms.end(); ++it, ++compIndex)
        {
            *it = new Matrix(_sepTask->magnitudeSpectraMatrix(0).rows(), 
                _sepTask->nrOfSpectra());
            for (unsigned int t = 0; t < _sepTask->nrOfSpectra(); ++t) {
                const Matrix& cs = _sepTask->magnitudeSpectraMatrix(t);
                (*it)->setColumn(t, cs.nthColumn(compIndex));
            }
        }
    }

    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();
    bool wienerRec = cfg.getBool("blissart.separation.export.wienerrec", 
                                    false)
    // Can only perform Wiener reconstruction if amplitude matrix is set.
                     && !_sepTask.isNull();

    // FIXME: Move to SeparationTask method.
    Poco::SharedPtr<Matrix> reconst;
    if (wienerRec) {
        reconst = new Matrix(_sepTask->phaseMatrix().rows(), 
                             _sepTask->phaseMatrix().cols());
        if (_sepTask->nrOfSpectra() > 1) {
            reconst->zero();
            Matrix hShifted = _sepTask->gainsMatrix();
            for (unsigned int t = 0; t < _sepTask->nrOfSpectra(); ++t) {
                reconst->add(_sepTask->magnitudeSpectraMatrix(t) * hShifted);
                hShifted.shiftColumnsRight();
            }
        }
        else {
            _sepTask->magnitudeSpectraMatrix(0).multWithMatrix(
                _sepTask->gainsMatrix(), reconst);
        }
    }

    Poco::Util::Application& app = BasicApplication::instance();
    DatabaseSubsystem &dbs = app.getSubsystem<DatabaseSubsystem>();

    // Initialize the DataSet.
    createDataSet();
    cout << "Class 1\n";
    incTotalProgress(0.25f);

    // Classify.
    ResponsePtr response(dbs.getResponse(_responseID));
    if (response.isNull()) {
        ostringstream msg;
        msg << "No response with id " << _responseID;
        logger().fatal(msg.str());
        throw Poco::RuntimeException(msg.str());
    }
    DataSet trainingSet = dbs.getDataSet(response, FeatureSet::getStandardSet());
    string scalingMethod = app.config().
        getString("blissart.classification.scaling.method", "minmax");
    if (scalingMethod == "minmax" || scalingMethod == "musigma") {
        // FIXME: Avoid copying
        vector<DataSet> dataSets(2);
        dataSets[0] = trainingSet;
        dataSets[1] = _dataSet;
        if (scalingMethod == "minmax") {
            const double lower = app.config().getDouble(
                "blissart.classification.scaling.lower", -1.0);
            const double upper = app.config().getDouble(
                "blissart.classification.scaling.upper", 1.0);
            linearScaleMinMax(dataSets, lower, upper);
        }
        else {
            const double mu = app.config().getDouble(
                "blissart.classification.scaling.mu", 0.0);
            const double sigma = app.config().getDouble(
                "blissart.classification.scaling.sigma", 1.0);
            linearScaleMuSigma(dataSets, mu, sigma);
        }
        trainingSet = dataSets[0];
        _dataSet = dataSets[1];
    }
    SVMModel model(trainingSet);
    model.classify(_dataSet);
    cout << "Class 2\n";
    incTotalProgress(0.25f);

    // Build a map of the classes and spectra.
    for (unsigned int i = 0; i < _componentSpectrograms.size(); ++i) {
        int cl;
        if (_presetClassLabels.find(i) == _presetClassLabels.end()) {
            cl = _dataSet[i].predictedClassLabel;
        }
        else {
            cl = _presetClassLabels[i];
        }

        Matrix* target;
        map<int, Matrix*>::const_iterator it = _spectraMap.find(cl);
        if (it == _spectraMap.end()) {
            target = new Matrix(_componentSpectrograms[0]->rows(), 
                                _gainsMatrix->cols(),
                                generators::zero);
            _spectraMap[cl] = target;
        } 
        else {
            target = it->second;
        }
        RowVector componentGains = _gainsMatrix->nthRow(i);
        for (unsigned int t = 0; t < _componentSpectrograms[i]->cols(); ++t) {
            target->add(_componentSpectrograms[i]->nthColumn(t) * componentGains);
            componentGains.shiftRight();
        }
    }

    if (wienerRec) {
        //logger().information("Aha");
        for (map<int, Matrix*>::iterator it = _spectraMap.begin();
             it != _spectraMap.end(); ++it)
        {
            it->second->elementWiseDivision(*reconst, it->second);
            it->second->elementWiseMultiplication(_sepTask->amplitudeMatrix(), 
                                                  it->second);
        }
    }

    cout << "Class 3\n";
    incTotalProgress(0.25f);

    exportAsWav();
    cout << "Class 4\n";
    incTotalProgress(0.25f);
}


void ClassificationTask::exportAsWav()
{
    DatabaseSubsystem &dbs =
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();

    for (map<int, Matrix*>::const_iterator it = _spectraMap.begin();
        it != _spectraMap.end(); it++)
    {
        // If prefix is "filename.wav", output files are named subsequently as
        // "filename_<ClassLabel1>.wav", ...
        const string labelText = dbs.getLabel(it->first)->text;
        Poco::Path file(_fileName);
        file.setBaseName(file.getBaseName() + "_" + labelText);
        // We're exporting .WAV, hence:
        file.setExtension("wav");

        Poco::SharedPtr<AudioData> pAd;
        BasicApplication::lockFFTW();
        try {
            pAd = AudioData::fromSpectrogram(*(it->second),
                                             *_phaseMatrix,
                                             &SqHannFunction, 
                                             _windowSize, 
                                             _overlap,
                                             _sampleRate);
        }
        catch (...) {
            BasicApplication::unlockFFTW();
            throw;
        }
        BasicApplication::unlockFFTW();

        logger().debug(nameAndTaskID() + " writing " + file.toString() + ".");
        WaveEncoder::saveAsWav(pAd->getChannel(0),
                               pAd->nrOfSamples(),
                               _sampleRate, 1, file.toString());
    }
}


void ClassificationTask::createDataSet()
{
    _dataSet.clear();
    for (unsigned int i = 0; i < _componentSpectrograms.size(); ++i) {
        DataPoint dp;
        FeatureExtractor::FeatureMap features;
        features = _extractor.extract(DataDescriptor::Spectrum,
            *_componentSpectrograms[i]);
        dp.components.insert(features.begin(), features.end());
        features = _extractor.extract(DataDescriptor::Gains,
            _gainsMatrix->nthRow(i));
        dp.components.insert(features.begin(), features.end());
        _dataSet.push_back(dp);
    }
}


} // namespace blissart
