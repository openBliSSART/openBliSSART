//
// $Id: SeparationTask.cpp 895 2009-07-06 13:38:27Z felix $
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


#include <blissart/SeparationTask.h>
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
// XXX: Is there a better place for Mel spectrum stuff?
#include <blissart/feature/mfcc.h>

#include <Poco/NumberFormatter.h>
#include <Poco/Util/LayeredConfiguration.h>

#include <cmath>


using namespace blissart::audio;
using namespace blissart::linalg;
using namespace std;


namespace blissart {


SeparationTask::SeparationTask(const SeparationMethod sepMethod,
                 const std::string &typeIdentifier,
                 SeparationTask::DataKind dataKind,
                 const std::string &fileName,
                 unsigned int nrOfComponents, unsigned int nrOfSpectra,
                 unsigned int maxIterations,
                 double epsilon, bool isVolatile) :
    FTTask(typeIdentifier, fileName),
    _separationMethod(sepMethod),
    _dataKind(dataKind),
    _nrOfComponents(nrOfComponents),
    _nrOfSpectra(nrOfSpectra),
    _maxIterations(maxIterations),
    _epsilon(epsilon),
    _isVolatile(isVolatile),
    _myUniqueID(retrieveUniqueID())
{
}


SeparationTask::~SeparationTask()
{
}


void SeparationTask::runTask()
{
    // Adapt Increase maxSteps as neccessary.
    incMaxProgress(0.2f);
    if (!_isVolatile)
        incMaxProgress(0.1f);
    if (!_exportPrefix.empty() && _dataKind == MagnitudeSpectrum)
        incMaxProgress(0.1f);
    registerTask(_myUniqueID, 1);

    // From now on, perform periodical checks to see if the task has been
    // cancelled.
    do {
        readAudioFile();
        incTotalProgress(0.1f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Compute the spectrogram.
        computeSpectrogram();
        if (_dataKind == MelSpectrum) {
            int nBanks = BasicApplication::instance().config().
                getInt("blissart.global.mel_bands", 26);
            Matrix *mel =
                feature::melSpectrum(amplitudeMatrix(), sampleRate(), nBanks);
            replaceAmplitudeMatrix(mel);
        }
        incTotalProgress(0.1f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Raise hell ;-)
        initialize();
        performSeparation();
        setTaskProgress(_myUniqueID, 1, 1.0f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Store the components. We don't want cancellation during the storage
        // process.
        if (!_isVolatile) {
            storeComponents();
            incTotalProgress(0.1f);
        }

        // The original amplitude matrix isn't needed anymore. Free some memory.
        deleteAmplitudeMatrix();

        // Mandatory check.
        if (isCancelled())
            break;

        if (!_exportPrefix.empty() && _dataKind == MagnitudeSpectrum) {
            exportComponents();
            incTotalProgress(0.1f);
        }
    } while (false);

    // The original amplitude matrix isn't needed anymore. Free some memory.
    // This has to be done again since it is possible that due to user
    // cancellation the upper do-while block was left before the first
    // free-attempt.
    deleteAmplitudeMatrix();
}


void SeparationTask::storeComponents() const
{
    debug_assert(&phaseMatrix() && &magnitudeSpectraMatrix(0));

    DatabaseSubsystem &dbs =
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    StorageSubsystem &sts =
        BasicApplication::instance().getSubsystem<StorageSubsystem>();

    logger().debug(nameAndTaskID() + " storing the components.");

    // Store a Process entity in the database. We summarize SEP and FFT
    // for the sake of simplicity...
    ProcessPtr newProcess;
    switch (_separationMethod) {
    case NMF:
        newProcess = new Process("NMF", fileName(), sampleRate());
        break;

    case NMD:
        newProcess = new Process("NMD", fileName(), sampleRate());
        break;

    default:
        throw Poco::NotImplementedException("Unknown separation method!");
    }
    newProcess->setWindowFunction(windowFunction());
    newProcess->setOverlap(overlap());
    newProcess->setWindowSize(windowSize());
    newProcess->parameters["dataKind"] = _dataKind == MagnitudeSpectrum ?
        "Magnitude spectrum" : "Mel spectrum";
    if (_dataKind == MelSpectrum) {
        newProcess->parameters["banks"] = BasicApplication::instance().
            config().getInt("blissart.global.mel_bands", 26);
    }
    newProcess->parameters["maxSteps"]   = Poco::NumberFormatter::format(_maxIterations);
    newProcess->parameters["epsilon"]    = Poco::NumberFormatter::format(_epsilon);
    newProcess->parameters["components"] = Poco::NumberFormatter::format(_nrOfComponents);
    newProcess->parameters["spectra"]    = Poco::NumberFormatter::format(_nrOfSpectra);
    dbs.createProcess(newProcess);

    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();
    int phaseMatrixID = 0, magnitudeMatrixID = 0;

    // Create a DataDescriptor for the phase matrix and save it,
    // so that we can reconstruct audio files from the separated components later.
    if (cfg.getBool("blissart.separation.storage.phasematrix", true)) {
        DataDescriptorPtr phaseMatrixDescr = new DataDescriptor;
        phaseMatrixDescr->type = DataDescriptor::PhaseMatrix;
        phaseMatrixDescr->processID = newProcess->processID;
        dbs.createDataDescriptor(phaseMatrixDescr);
        phaseMatrixID = phaseMatrixDescr->descrID;
        sts.store(phaseMatrix(), phaseMatrixDescr);
    }

    // Also save the amplitude matrix.
    if (_dataKind == MelSpectrum ||
        cfg.getBool("blissart.separation.storage.magnitudematrix", false)) {
        DataDescriptorPtr magnMatrixDescr = new DataDescriptor;
        magnMatrixDescr->type = _dataKind == MagnitudeSpectrum ?
            DataDescriptor::MagnitudeMatrix : DataDescriptor::MelMatrix;
        magnMatrixDescr->processID = newProcess->processID;
        dbs.createDataDescriptor(magnMatrixDescr);
        magnitudeMatrixID = magnMatrixDescr->descrID;
        sts.store(amplitudeMatrix(), magnMatrixDescr);
    }

    // Store the components.
    for (unsigned int i = 0; i < _nrOfComponents; i++) {
        // Create a ClassificationObject for the current spectrum- and gains-
        // matrix as well as the phase- and amplitude-matrix.
        ClassificationObjectPtr componentObject = new ClassificationObject;
        if (phaseMatrixID > 0) {
            componentObject->descrIDs.insert(phaseMatrixID);
        }
        if (magnitudeMatrixID > 0) {
            componentObject->descrIDs.insert(magnitudeMatrixID);
        }

        // Spectrum.
        Matrix componentSpectrogram(magnitudeSpectraMatrix(0).rows(),
            _nrOfSpectra);
        for (unsigned int t = 0; t < _nrOfSpectra; t++) {
            componentSpectrogram.setColumn(t,
                magnitudeSpectraMatrix(t).nthColumn(i));
        }
        DataDescriptorPtr spectrumDescr = new DataDescriptor;
        spectrumDescr->type = _dataKind == MagnitudeSpectrum ?
            DataDescriptor::Spectrum : DataDescriptor::MelSpectrum;
        spectrumDescr->index = i;
        spectrumDescr->processID = newProcess->processID;
        dbs.createDataDescriptor(spectrumDescr);
        sts.store(componentSpectrogram, spectrumDescr);
        componentObject->descrIDs.insert(spectrumDescr->descrID);

        // Gains.
        DataDescriptorPtr gainsDescr = new DataDescriptor;
        gainsDescr->type = DataDescriptor::Gains;
        gainsDescr->index = i;
        gainsDescr->processID = newProcess->processID;
        dbs.createDataDescriptor(gainsDescr);
        sts.store(gainsMatrix().nthRow(i), gainsDescr);

        componentObject->descrIDs.insert(gainsDescr->descrID);
        switch (_separationMethod) {
        case NMF:
            componentObject->type = ClassificationObject::NMFComponent;
            break;
        case NMD:
            componentObject->type = ClassificationObject::NMDComponent;
            break;
        default:
            throw Poco::NotImplementedException("Unknown separation method.");
        }
        dbs.createClassificationObject(componentObject);
    }
}


void SeparationTask::exportComponents() const
{
    debug_assert(&phaseMatrix() &&
                 &magnitudeSpectraMatrix(0) &&
                 &gainsMatrix() &&
                 !_exportPrefix.empty());

    logger().debug(nameAndTaskID() + " exporting the components.");

    // Store the components.
    for (unsigned int i = 0; i < _nrOfComponents; i++) {
        // Compute the component's magnitude spectrum.
        Matrix magnitudeSpectrum(magnitudeSpectraMatrix(0).rows(),
            gainsMatrix().cols());
        // NMD case
        if (_nrOfSpectra > 1) {
            magnitudeSpectrum.zero();
            RowVector componentGains = gainsMatrix().nthRow(i);
            for (unsigned int t = 0; t < _nrOfSpectra; t++) {
                ColVector componentSpectrum = magnitudeSpectraMatrix(t).nthColumn(i);
                magnitudeSpectrum.add(componentSpectrum * componentGains);
                componentGains.shiftRight();
            }
        }
        // NMF case (separated for efficiency)
        else {
            ColVector componentSpectrum = magnitudeSpectraMatrix(0).nthColumn(i);
            RowVector componentGains = gainsMatrix().nthRow(i);
            magnitudeSpectrum = componentSpectrum * componentGains;
        }

        // Create an AudioData object.
        Poco::SharedPtr<AudioData> pAd =
                AudioData::fromSpectrogram(magnitudeSpectrum,
                                           phaseMatrix(),
                                           windowFunction(),
                                           windowSize(),
                                           overlap(),
                                           sampleRate());

        // Construct the filename.
        const int numDigits = (int)(1 + log10f((float)_nrOfComponents));
        stringstream ss;
        ss << _exportPrefix << '_' << taskID() << '_'
           << setfill('0') << setw(numDigits) << i << ".wav";

        // Eventually export the component.
        if (!WaveEncoder::saveAsWav(*pAd, ss.str()))
            throw runtime_error("Couldn't export to " + ss.str() + "!");
    }
}


void SeparationTask::setSeparationProgress(float progress)
{
    debug_assert(progress >= 0 && progress <= 1);
    setTaskProgress(_myUniqueID, 1, progress);
}


void SeparationTask::
setInitializationObjects(const std::vector<ClassificationObjectPtr>& objects,
                         bool constant)
{
    if (objects.size() > _nrOfComponents) {
        throw Poco::InvalidArgumentException(
            "Too many classification objects for initialization."
        );
    }

    _initObjects = objects;
    _constantInitializedComponentsSpectra = constant;
}


} // namespace blissart
