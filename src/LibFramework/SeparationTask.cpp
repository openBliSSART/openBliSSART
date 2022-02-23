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


#include <blissart/SeparationTask.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/ClassificationObject.h>
#include <blissart/DataDescriptor.h>
#include <blissart/Process.h>
#include <blissart/BinaryReader.h>
#include <blissart/GnuplotWriter.h>
#include <blissart/HTKWriter.h>
#include <blissart/transforms/MelFilterTransform.h>
#include <blissart/nmf/randomGenerator.h>

#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>

#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>

#include <Poco/NumberFormatter.h>
#include <Poco/Path.h>
#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/LogStream.h>

#include <cmath>
#include <fstream>
#include <ctime>


using namespace blissart::audio;
using namespace blissart::linalg;
using namespace std;


namespace blissart {


SeparationTask::SeparationTask(const SeparationMethod sepMethod,
                 const std::string &typeIdentifier,
                 const std::string &fileName,
                 unsigned int nrOfComponents, unsigned int nrOfSpectra,
                 unsigned int maxIterations,
                 double epsilon, bool isVolatile) :
    FTTask(typeIdentifier, fileName, isVolatile),
    _separationMethod(sepMethod),
    _nrOfComponents(nrOfComponents),
    _nrOfSpectra(nrOfSpectra),
    _constantInitializedComponentsSpectra(true),
    _constantInitializedGains(false),
    _genFunc(nmf::gaussianRandomGenerator),
    _maxIterations(maxIterations),
    _epsilon(epsilon),
    _relativeError(0.0),
    _exportComponents(false),
    _exportSpectra(false),
    _exportGains(false),
    _myUniqueID(retrieveUniqueID())
{
}


SeparationTask::~SeparationTask()
{
}


void SeparationTask::runTask()
{
    // Take into account loading audio and computing FFT.
    incMaxProgress(0.2f);
    // Take into account additional transformations.
    incMaxProgress(0.05f * transforms().size());
    // Take into account storage of components.
    if (!isVolatile())
        incMaxProgress(0.1f);
    // Take into account export of components / matrices.
    if (_exportComponents)
        incMaxProgress(0.1f);
    if (_exportSpectrogram || _exportSpectra || _exportGains)
        incMaxProgress(0.1f);

    registerTask(_myUniqueID, 1);

    // From now on, perform periodical checks to see if the task has been
    // cancelled.
    do {
        bool pmf = processMatrixFile();
        if (pmf) {
            logger().debug("Processing matrix file: " + fileName());
            Poco::Util::LayeredConfiguration& cfg =
                BasicApplication::instance().config();
            _sampleRate = cfg.getInt("blissart.audio.sampleRate", 44100);
        }
        else {
            logger().debug("Processing audio file: " + fileName());
            readAudioFile();
        }
        incTotalProgress(0.1f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Compute the spectrogram.
        if (pmf) {
            Matrix *inputMatrix = new Matrix(fileName());
            replaceAmplitudeMatrix(inputMatrix);
            if (_phaseMatrixFile.empty())
                _phaseMatrix = new Matrix(inputMatrix->rows(), inputMatrix->cols(), generators::zero);
            else
                _phaseMatrix = new Matrix(_phaseMatrixFile);
        }
        else {
            computeSpectrogram();
        }
        incTotalProgress(0.1f);

        // Mandatory check.
        if (isCancelled())
            break;

        doAdditionalTransformations();
        // Mandatory check.
        if (isCancelled())
            break;

        // Raise hell ;-)
        initialize();
        clock_t start = clock();
        performSeparation();
        clock_t end = clock();
        double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
        logger().debug(nameAndTaskID() + ": separation and reconstruction took " + Poco::NumberFormatter::format(elapsed, 2) + " seconds");

        if (_computeRelativeError) {
            computeRelativeError();
        }
        setTaskProgress(_myUniqueID, 1, 1.0f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Store the components. We don't want cancellation during the storage
        // process.
        if (!isVolatile()) {
            storeComponents();
            incTotalProgress(0.1f);
        }

        // The original amplitude matrix isn't needed anymore.
        // Hence, free some memory.
        // deleteAmplitudeMatrix();

        // Mandatory check.
        if (isCancelled())
            break;

        // Export components as audio files, if desired.
        if (_exportComponents) {
            exportComponents();
            incTotalProgress(0.1f);
        }

        // Check again.
        if (isCancelled())
            break;

        // Export separation matrices, if desired.
        if (_exportSpectrogram || _exportSpectra || _exportGains) {
            exportMatrices();
            incTotalProgress(0.1f);
        }

    } while (false);

    // The original amplitude matrix isn't needed anymore. Free some memory.
    // This has to be done again since it is possible that due to user
    // cancellation the upper do-while block was left before the first
    // free-attempt.
    //deleteAmplitudeMatrix();
}


void SeparationTask::setProcessParameters(ProcessPtr process) const
{
    FTTask::setProcessParameters(process);
    process->parameters["maxSteps"]   = Poco::NumberFormatter::format(_maxIterations);
    process->parameters["epsilon"]    = Process::formatDouble(_epsilon);
    process->parameters["components"] = Poco::NumberFormatter::format(_nrOfComponents);
    process->parameters["spectra"]    = Poco::NumberFormatter::format(_nrOfSpectra);
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
    case NMD:
        newProcess = new Process("NMD", fileName(), sampleRate());
        break;
    default:
        throw Poco::NotImplementedException("Unknown separation method!");
    }
    setProcessParameters(newProcess);
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
    if (cfg.getBool("blissart.separation.storage.magnitudematrix", false)) {
        DataDescriptorPtr magnMatrixDescr = new DataDescriptor;
        magnMatrixDescr->type = DataDescriptor::MagnitudeMatrix;
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
        spectrumDescr->type = DataDescriptor::Spectrum;
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
        case NMD:
            componentObject->type = ClassificationObject::NMDComponent;
            break;
        default:
            throw Poco::NotImplementedException("Unknown separation method.");
        }
        dbs.createClassificationObject(componentObject);
    }
}


Poco::SharedPtr<Matrix>
SeparationTask::revertTransforms(Poco::SharedPtr<Matrix> spectrogram) const
{
    // Revert any transformations, in reverse order.
    vector<MatrixTransform*>::const_reverse_iterator tflast(transforms().end());
    vector<MatrixTransform*>::const_reverse_iterator tffirst(transforms().begin());
    for (vector<MatrixTransform*>::const_reverse_iterator rit = tflast;
         rit != tffirst; ++rit) // YES, it's operator ++ ;-)
    {
        if (string((*rit)->name()) == "Mel filter") {
            int bins = windowSize() * sampleRate() / 1000 / 2 + 1;
            ((transforms::MelFilterTransform*)(*rit))->setBins(bins);
        }
        // Let Poco::SharedPtr do the dirty work of pointer handling!
        spectrogram = (*rit)->inverseTransform(spectrogram);
    }
    return spectrogram;
}


// XXX: maybe merge with AudioObject code?

void SeparationTask::spectrogramToAudioFile(
    Poco::SharedPtr<Matrix> magnitudeSpectrum,
    const std::string& outputFile) const
{
    // Create an AudioData object.
    BasicApplication::lockFFTW();
    Poco::SharedPtr<AudioData> pAd;
    try {
        logger().debug("Applying inverse FFT.");
        pAd = AudioData::fromSpectrogram(*magnitudeSpectrum,
                                         phaseMatrix(),
                                         windowFunction(),
                                         windowSize(),
                                         overlap(),
                                         sampleRate());
    }
    catch (...) {
        BasicApplication::unlockFFTW();
        throw;
    }
    BasicApplication::unlockFFTW();

    // Eventually export the component.
    if (!WaveEncoder::saveAsWav(*pAd, outputFile))
        throw runtime_error("Couldn't export to " + outputFile + "!");
}


void SeparationTask::exportComponents() const
{
    debug_assert(&phaseMatrix() &&
                 &magnitudeSpectraMatrix(0) &&
                 &gainsMatrix());

    logger().debug(nameAndTaskID() + " exporting the components.");

    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();

    // Determine whether to export components as WAV and/or as matrix files.
    bool exportAsMatrix = cfg.getBool("blissart.separation.export.matrix",
                                      false);
    if (exportAsMatrix)
        logger().debug("Exporting components as matrix file(s).");
    bool exportAsWave = cfg.getBool("blissart.separation.export.wave",
                                      true);
    if (exportAsWave)
        logger().debug("Exporting components as waveform file(s).");

    // Determine whether to use wiener reconstruction.
    bool wienerRec = cfg.getBool("blissart.separation.export.wienerrec", 
                                    true);

    // Compute the reconstructed matrix (WH) in case of wiener reconstruction.
    Poco::SharedPtr<Matrix> reconst;
    if (wienerRec) {
        //reconst = new Matrix(phaseMatrix().rows(), phaseMatrix().cols());
        reconst = new Matrix(magnitudeSpectraMatrix(0).rows(),
            gainsMatrix().cols());
        if (_nrOfSpectra > 1) {
            reconst->zero();
            Matrix hShifted = gainsMatrix();
            for (unsigned int t = 0; t < _nrOfSpectra; ++t) {
                reconst->add(magnitudeSpectraMatrix(t) * hShifted);
                hShifted.shiftColumnsRight();
            }
        }
        else {
            magnitudeSpectraMatrix(0).multWithMatrix(gainsMatrix(), reconst);
        }
        // revert transform to reconst
        reconst = revertTransforms(reconst);
    }

    // Retrieve desired component indices.
    vector<vector<int> > compIndices = _exportComponentIndices;
    if (compIndices.empty()) {
        vector<int> compIndicesSource;
        for (int i = 1; i <= _nrOfComponents; i++) {
            compIndicesSource.push_back(i);
        }
        compIndices.push_back(compIndicesSource);
    }

    // Reconstruct components and mix, if desired.
    int sourceIndex = 1;
    for (vector<vector<int> >::const_iterator sourceIt = compIndices.begin();
         sourceIt != compIndices.end(); ++sourceIt, ++sourceIndex)
    {
        // Holds the mixed spectrogram if mixing is desired.
        Poco::SharedPtr<Matrix> mixedSpectrogram;

        logger().debug("Exporting components for source #" +
            Poco::NumberFormatter::format(sourceIndex));
        for (vector<int>::const_iterator it = sourceIt->begin();
            it != sourceIt->end(); ++it)
        {
            if (*it < 1 || *it > _nrOfComponents) {
                logger().error(nameAndTaskID() + ": invalid component index: " +
                    Poco::NumberFormatter::format(*it));
                continue;
            }
            int i = *it - 1;
            logger().debug(nameAndTaskID() + " exporting component #" +
                Poco::NumberFormatter::format(i));
            // Compute the component's magnitude spectrum.
            Poco::SharedPtr<Matrix> magnitudeSpectrum = new Matrix(
                magnitudeSpectraMatrix(0).rows(),
                gainsMatrix().cols());
            // NMD case
            if (_nrOfSpectra > 1) {
                magnitudeSpectrum->zero();
                RowVector componentGains = gainsMatrix().nthRow(i);
                for (unsigned int t = 0; t < _nrOfSpectra; t++) {
                    ColVector componentSpectrum = magnitudeSpectraMatrix(t).nthColumn(i);
                    magnitudeSpectrum->add(componentSpectrum * componentGains);
                    componentGains.shiftRight();
                }
            }
            // "NMF" case (separated for efficiency)
            else {
                ColVector componentSpectrum = magnitudeSpectraMatrix(0).nthColumn(i);
                RowVector componentGains = gainsMatrix().nthRow(i);
                *magnitudeSpectrum = componentSpectrum * componentGains;
            }

            // revert transformation to component spectrogram
            magnitudeSpectrum = revertTransforms(magnitudeSpectrum);

            if (wienerRec) {
                // (Component/Whole) reconstruction
                reconst->floor(1e-6);
                magnitudeSpectrum->elementWiseDivision(*reconst, magnitudeSpectrum);
                // Use as filter for original spectrogram
                magnitudeSpectrum->elementWiseMultiplication(ftMagMatrix(),
                                                       magnitudeSpectrum);
            }

            // Mix the components to a single spectrogram that is exported after
            // the loop.
            if (_mixExportedComponents) {
                if (mixedSpectrogram.isNull()) {
                    mixedSpectrogram = new Matrix(magnitudeSpectrum->rows(), 
                        magnitudeSpectrum->cols());
                    mixedSpectrogram->zero();
                }
                mixedSpectrogram->add(*magnitudeSpectrum);
            }
            // Export components individually.
            else {
                // Construct the filename.
                string prefix = getExportPrefix();
                const int numDigits = (int)(1 + log10f((float)_nrOfComponents));
                stringstream ss;
                ss << prefix;
                if (compIndices.size() > 1) {
                    const int numDigitsS = (int)(1 + log10f((float)compIndices.size()));
                    ss << '_' << setfill('0') << setw(numDigitsS) << sourceIndex;
                }
                ss << '_' << setfill('0') << setw(numDigits) << *it;

                if (exportAsMatrix) {
                    magnitudeSpectrum->dump(ss.str() + ".dat");
                }
                if (exportAsWave) {
                    // Convert component spectrogram to time signal and save it as WAV.
                    spectrogramToAudioFile(magnitudeSpectrum, ss.str() + ".wav");
                }
            }
        }
        // Export the mixed source spectrogram to a single audio file.
        if (_mixExportedComponents) {
            // Construct the filename.
            stringstream ss;
            ss << getExportPrefix();
            const int numDigitsS = (int)(1 + log10f((float)compIndices.size()));
            ss << "_source" << setfill('0') << setw(numDigitsS) << sourceIndex;
            // Convert component spectrogram to time signal and save it as WAV.
            string filename = ss.str();
            logger().debug(nameAndTaskID() + ": creating mixed audio file " +
                filename);
            if (exportAsMatrix) {
                mixedSpectrogram->dump(filename + ".dat");
            }
            if (exportAsWave) {
                spectrogramToAudioFile(mixedSpectrogram, filename + ".wav");
            }
        }
    }

}


void SeparationTask::exportMatrices() const
{
    Poco::Util::LayeredConfiguration& cfg =
        BasicApplication::instance().config();
    bool useGnuplotFormat = 
        cfg.getString("blissart.separation.export.format", "bin")
        .substr(0, 3) == "gnu";
    bool useHTKFormat = 
        cfg.getString("blissart.separation.export.format", "bin")
        .substr(0, 3) == "htk";
    bool asTensor =
        cfg.getBool("blissart.separation.export.format.bin.asTensor", true);

    // Construct the prefix.
    string prefix = getExportPrefix();
    if (_exportSpectra) {
        if (useGnuplotFormat || useHTKFormat || !asTensor) {
            for (unsigned int i = 0; i < _nrOfSpectra; ++i) {
                const int numDigits = (int)(1 + log10f((float)_nrOfSpectra));
                stringstream ss;
                ss << prefix /* << '_' << taskID() */ << "_W_"
                   << setfill('0') << setw(numDigits) << i << ".dat";
                // TODO: error handling?
                logger().debug("Writing to " + ss.str());
                if (useGnuplotFormat)
                    GnuplotWriter::writeMatrixGnuplot(magnitudeSpectraMatrix(i),
                                                      ss.str(), false);
                else if (useHTKFormat)
                    exportMatrixHTK(magnitudeSpectraMatrix(i), ss.str());
                else
                    magnitudeSpectraMatrix(i).dump(ss.str());
            }
        }
        else {
            vector<const Matrix*> mv;
            for (unsigned int i = 0; i < _nrOfSpectra; ++i) {
                mv.push_back(&(magnitudeSpectraMatrix(i)));
            }
            Matrix::arrayToFile(mv, prefix + "_W.dat");
        }
    }
    if (_exportSpectrogram) {
        exportSpectrogram();
    }
    if (_exportGains) {
        stringstream ss;
        ss << prefix /* << '_' << taskID() */ << "_H.dat";
        logger().debug("Writing to " + ss.str());
        if (useGnuplotFormat) 
            GnuplotWriter::writeMatrixGnuplot(gainsMatrix(), ss.str(), true);
        else if (useHTKFormat) 
            exportMatrixHTK(gainsMatrix(), ss.str());
        else 
            // BliSSART binary format
            gainsMatrix().dump(ss.str());
    }
}


void SeparationTask::exportMatrixHTK(const blissart::linalg::Matrix& m,
                                     const std::string &filename) const
{
    ofstream mos(filename.c_str());
    // calculate framerate in 100ns (HTK unit)
    int framerate = (int) ((double)windowSize() * (1.0 - overlap())) 
                    * 10000;  // 1 ms = 10^4 * 100ns
    HTKWriter::writeMatrix(mos, m, framerate);
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


void SeparationTask::
setInitializationMatrices(const std::vector<std::string>& matrices,
                         bool constant)
{
    if (matrices.size() > _nrOfComponents) {
        throw Poco::InvalidArgumentException(
            "Too many classification objects for initialization."
        );
    }

    _initMatrices = matrices;
    _constantInitializedComponentsSpectra = constant;
}


void SeparationTask::
setGainsInitializationMatrix(const std::string& filename,
                             bool constant)
{
    _gainsInitMatrix = filename;
    _constantInitializedGains = constant;
}


bool SeparationTask::processMatrixFile() const
{
    unsigned int tmp = 0;
    ifstream mifs(fileName().c_str());
    BinaryReader br(mifs, BinaryReader::LittleEndian);
    br >> tmp;
    return (tmp == 2);
}


} // namespace blissart
