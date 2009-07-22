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


#include <blissart/FTTask.h>
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
               const std::string &fileName) :
    BasicTask(typeIdentifier),
    _fileName(fileName),
    _audioData(0),
    _sampleRate(0),
    _amplitudeMatrix(0),
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
    if (_amplitudeMatrix) {
        delete _amplitudeMatrix;
        _amplitudeMatrix = 0;
    }
    if (_phaseMatrix) {
        delete _phaseMatrix;
        _phaseMatrix = 0;
    }
}


void FTTask::runTask()
{
    // Steps: Read audio, compute spectrogram, save it.
    incMaxProgress(3.0f);

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

        // Store the matrices.
        storeComponents();
        incTotalProgress(1.0f);

        // Mandatory check.
        if (isCancelled())
            break;
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

    logger().debug(nameAndTaskID() + " computing spectrogram.");
    pair<Matrix*, Matrix*> spectrogram = _audioData->
        computeSpectrogram(_windowFunction, _windowSize, _overlap, 0,
                           _zeroPadding, _removeDC);
    _amplitudeMatrix = spectrogram.first;
    _phaseMatrix = spectrogram.second;

    BasicApplication::unlockFFTW();
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
    process->setWindowFunction(_windowFunction);
    process->setOverlap(_overlap);
    process->setWindowSize(_windowSize);
    dbs.createProcess(process);

    // Create a DataDescriptor for the magnitude matrix and save it.
    DataDescriptorPtr magnMatrixDescr = new DataDescriptor;
    magnMatrixDescr->type = DataDescriptor::MagnitudeMatrix;
    magnMatrixDescr->processID = process->processID;
    dbs.createDataDescriptor(magnMatrixDescr);
    sts.store(*_amplitudeMatrix, magnMatrixDescr);

    // Create a DataDescriptor for the phase matrix and save it.
    DataDescriptorPtr phaseMatrixDescr = new DataDescriptor;
    phaseMatrixDescr->type = DataDescriptor::PhaseMatrix;
    phaseMatrixDescr->processID = process->processID;
    dbs.createDataDescriptor(phaseMatrixDescr);
    sts.store(*_phaseMatrix, phaseMatrixDescr);

    // Create a ClassificationObject for the spectrogram.
    ClassificationObjectPtr clObj = new ClassificationObject;
    clObj->type = ClassificationObject::Spectrogram;
    clObj->descrIDs.insert(magnMatrixDescr->descrID);
    clObj->descrIDs.insert(phaseMatrixDescr->descrID);
    dbs.createClassificationObject(clObj);
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
