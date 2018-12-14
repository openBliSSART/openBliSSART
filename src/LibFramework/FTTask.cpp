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


#include <blissart/FTTask.h>
#include <blissart/transforms/SpectralSubtractionTransform.h>
#include <blissart/transforms/PowerTransform.h>
#include <blissart/transforms/SlidingWindowTransform.h>
#include <blissart/transforms/MelFilterTransform.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/ClassificationObject.h>
#include <blissart/DataDescriptor.h>
#include <blissart/Process.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>

#include <Poco/NumberFormatter.h>
#include <Poco/Util/LayeredConfiguration.h>

#include <cmath>


using namespace blissart::audio;
using namespace blissart::linalg;
using namespace std;


namespace blissart {


FTTask::FTTask(const std::string& typeIdentifier,
               const std::string &fileName,
               bool isVolatile) :
    BasicTask(typeIdentifier),
    _fileName(fileName),
    _isVolatile(isVolatile),
    _audioData(0),
    _sampleRate(0),
    _amplitudeMatrix(0),
    _ftMagMatrix(0),
    _phaseMatrix(0)
{
    logger().debug(nameAndTaskID() + " reading configuration.");

    // Get audio processing and FFT parameters from configuration.
    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();
    _windowFunction = windowFunctionForShortName(
        cfg.getString("blissart.fft.windowfunction", "sqhann"));
    _windowSize = cfg.getInt("blissart.fft.windowsize", 25);
    _overlap = cfg.getDouble("blissart.fft.overlap", 0.5);
    _preemphasisCoeff = cfg.getDouble("blissart.audio.preemphasis", 0.0);
    _zeroPadding = cfg.getBool("blissart.fft.zeropadding", false);
    _removeDC = cfg.getBool("blissart.audio.remove_dc", false);
    _reduceMids = cfg.getBool("blissart.audio.reduce_mids", false);

}


FTTask::~FTTask()
{
    if (_audioData) {
        delete _audioData;
        _audioData = 0;
    }
    if (_ftMagMatrix && _ftMagMatrix != _amplitudeMatrix) {
        delete _ftMagMatrix;
        _ftMagMatrix = 0;
    }
    if (_amplitudeMatrix) {
        delete _amplitudeMatrix;
        _amplitudeMatrix = 0;
    }
    if (_phaseMatrix) {
        delete _phaseMatrix;
        _phaseMatrix = 0;
    }
    for (vector<MatrixTransform*>::iterator it = _transforms.begin();
         it != _transforms.end(); ++it)
    {
        if (*it) delete *it;
    }
}


void FTTask::runTask()
{
    // Steps: Read audio, compute spectrogram, save it.
    incMaxProgress(2.0f);
    if (!isVolatile())
        incMaxProgress(1.0f);

    // Take into account additional transformations.
    incMaxProgress(0.5f * _transforms.size());

    // From now on, perform periodical checks to see if the task has been
    // cancelled.
    do {
        // Get the audio source, set the sample rate
        readAudioFile();
        incTotalProgress(1.0f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Compute the spectrogram.
        computeSpectrogram();
        incTotalProgress(1.0f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Additional transformations, if desired.
        doAdditionalTransformations();

        // Mandatory check.
        if (isCancelled())
            break;

        // Store the matrices.
        if (!isVolatile()) {
            storeComponents();
            incTotalProgress(1.0f);
        }

        // Mandatory check.
        if (isCancelled())
            break;

        if (_exportSpectrogram)
            exportSpectrogram();

    } while (false);
}


void FTTask::readAudioFile()
{
    logger().debug(nameAndTaskID() + " loading " + _fileName);
    _audioData = AudioData::fromFile(_fileName);
    _sampleRate = _audioData->sampleRate();
    // XXX: preemphasis before mid-reduction? (makes sense)
    // Preemphasis with coefficient 0 does nothing, but causes overhead
    // - thus, we only call preemphasize() if coefficient is > 0
    if (_preemphasisCoeff > 0.0) {
        _audioData->preemphasize(_preemphasisCoeff);
    }
    if (_audioData->nrOfChannels() > 1) {
        logger().debug(nameAndTaskID() + " reducing audio to 1 channel.");
        if (_reduceMids && _audioData->nrOfChannels() == 2) {
            _audioData->subRightFromLeftChannel();
        }
        else {
            _audioData->makeMono();
        }
    }
}


void FTTask::computeSpectrogram()
{
    debug_assert(!_amplitudeMatrix);

    BasicApplication::lockFFTW();
    pair<Matrix*, Matrix*> spectrogram;
    logger().debug(nameAndTaskID() + " computing spectrogram.");
    try {
        pair<Matrix*, Matrix*> spectrogram = _audioData->
            computeSpectrogram(_windowFunction, _windowSize, _overlap, 0,
                               _zeroPadding, _removeDC);
        _amplitudeMatrix = spectrogram.first;
        _phaseMatrix = spectrogram.second;
    }
    catch (...) {
        BasicApplication::unlockFFTW();
        throw;
    }
    BasicApplication::unlockFFTW();

    _ftMagMatrix = _amplitudeMatrix;

    // Add transformations if specified in the configuration.
    // Keep in mind that the transformations are executed in the order in
    // which they were added, thus the order of these statements is vital!
    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();
    if (cfg.getBool("blissart.fft.transformations.spectralSubtraction", false)) {
        addTransformation(new transforms::SpectralSubtractionTransform);
    }
    if (cfg.getBool("blissart.fft.transformations.powerSpectrum", false)) {
        addTransformation(new transforms::PowerTransform);
    }
    if (cfg.getBool("blissart.fft.transformations.melFilter", false)) {
        addTransformation(new transforms::MelFilterTransform(_sampleRate));
    }
    if (cfg.getBool("blissart.fft.transformations.slidingWindow", false)) {
        addTransformation(new transforms::SlidingWindowTransform);
    }
}


void FTTask::doAdditionalTransformations()
{
    // Backup FT magnitude matrix if transformations are applied afterwards.
    if (!_transforms.empty()) {
        _ftMagMatrix = new Matrix(*_amplitudeMatrix);
    }
    // Apply transformations in specified order.
    for (vector<MatrixTransform*>::const_iterator it = _transforms.begin();
         it != _transforms.end() && !isCancelled(); ++it)
    {
        Matrix *trResult = (*it)->transform(_amplitudeMatrix);
        if (trResult != _amplitudeMatrix) {
            replaceAmplitudeMatrix(trResult);
        }
        incTotalProgress(0.5f);
    }
}


void FTTask::setProcessParameters(ProcessPtr process) const
{
    process->setWindowFunction(_windowFunction);
    process->setOverlap(_overlap);
    process->setWindowSize(_windowSize);
    process->parameters["transformCount"] = 
        Poco::NumberFormatter::format(_transforms.size());
    int trIndex = 1;
    for (vector<MatrixTransform*>::const_iterator it = _transforms.begin();
         it != _transforms.end(); ++it, ++trIndex)
    {
        string paramName = "transform" + Poco::NumberFormatter::format(trIndex);
        process->parameters[paramName] = (*it)->name();
        MatrixTransform::TransformParameters tp = (*it)->getParameters();
        for (MatrixTransform::TransformParameters::const_iterator 
             pit = tp.begin(); pit != tp.end(); ++pit)
        {
            string tparamName = paramName + pit->first;
            process->parameters[tparamName] = pit->second;
        }
    }
}


void FTTask::storeComponents() const
{
    debug_assert(_phaseMatrix && _amplitudeMatrix);

    DatabaseSubsystem &dbs =
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    StorageSubsystem &sts =
        BasicApplication::instance().getSubsystem<StorageSubsystem>();

    logger().debug(nameAndTaskID() + " storing the matrices.");

    // Store a Process entity in the database.
    ProcessPtr process = new Process("FT", _fileName, _sampleRate);
    setProcessParameters(process);
    dbs.createProcess(process);

    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();
    int phaseMatrixID = 0, magnitudeMatrixID = 0;

    // Create a DataDescriptor for the phase matrix and save it.
    if (cfg.getBool("blissart.fft.storage.phasematrix", true)) {
        DataDescriptorPtr phaseMatrixDescr = new DataDescriptor;
        phaseMatrixDescr->type = DataDescriptor::PhaseMatrix;
        phaseMatrixDescr->processID = process->processID;
        dbs.createDataDescriptor(phaseMatrixDescr);
        phaseMatrixID = phaseMatrixDescr->descrID;
        sts.store(*_phaseMatrix, phaseMatrixDescr);
    }

    // Create a DataDescriptor for the magnitude matrix and save it.
    if (cfg.getBool("blissart.fft.storage.magnitudematrix", true)) {
        DataDescriptorPtr magnMatrixDescr = new DataDescriptor;
        magnMatrixDescr->type = DataDescriptor::MagnitudeMatrix;
        magnMatrixDescr->processID = process->processID;
        dbs.createDataDescriptor(magnMatrixDescr);
        magnitudeMatrixID = magnMatrixDescr->descrID;
        sts.store(*_amplitudeMatrix, magnMatrixDescr);
    }

    // Create a ClassificationObject for the spectrogram.
    ClassificationObjectPtr clObj = new ClassificationObject;
    clObj->type = ClassificationObject::Spectrogram;
    if (phaseMatrixID > 0) {
        clObj->descrIDs.insert(phaseMatrixID);
    }
    if (magnitudeMatrixID > 0) {
        clObj->descrIDs.insert(magnitudeMatrixID);
    }
    dbs.createClassificationObject(clObj);
}


void FTTask::exportSpectrogram() const
{
    if (_amplitudeMatrix) {
        string fileName = getExportPrefix() + "_V.dat";
        logger().debug("Writing to " + fileName);
        _amplitudeMatrix->dump(fileName);
    }
    if (_phaseMatrix) {
        Poco::Util::LayeredConfiguration& cfg =
            BasicApplication::instance().config();
        if (cfg.getBool("blissart.audio.export.phase", false)) {
            string fileName = getExportPrefix() + "_V_phase.dat";
            logger().debug("Writing to " + fileName);
            _phaseMatrix->dump(fileName);
        }
    }
}


string FTTask::getExportPrefix() const
{
    string prefix = _exportPrefix;
    if (prefix.empty()) {
        prefix = fileName().substr(0, fileName().find_last_of('.'));
    }
    else {
        Poco::Path tmp(fileName());
        tmp.makeFile();
        prefix += tmp.getBaseName();
    }
    return prefix;
}


void FTTask::deletePhaseMatrix()
{
    if (_phaseMatrix) {
        delete _phaseMatrix;
        _phaseMatrix = 0;
    }
}


void FTTask::replaceAmplitudeMatrix(Matrix* amplitudeMatrix)
{
    if (_amplitudeMatrix)
        delete _amplitudeMatrix;

    _amplitudeMatrix = amplitudeMatrix;
}


void FTTask::deleteAmplitudeMatrix()
{
    if (_amplitudeMatrix) {
        delete _amplitudeMatrix;
        _amplitudeMatrix = 0;
    }
}


} // namespace blissart
