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


#include <blissart/AudioObject.h>
#include <blissart/ProgressObserver.h>
#include <blissart/audio/AudioData.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/transforms/PowerTransform.h>
#include <Poco/NumberParser.h>
#include <Poco/NumberFormatter.h>
#include <Poco/Exception.h>



using namespace blissart::audio;
using namespace blissart::linalg;
using namespace std;


namespace blissart {


AudioData* AudioObject::getAudioObject(ClassificationObjectPtr clo,
                                       ProgressObserver* obs)
{
    cout << "\nGetAudioObject\n";
    DatabaseSubsystem& dbs = BasicApplication::instance().
        getSubsystem<DatabaseSubsystem>();
    StorageSubsystem& sts = BasicApplication::instance().
        getSubsystem<StorageSubsystem>();

    // Get all data descriptors associated with the underlying classification
    // object.
    vector<DataDescriptorPtr> dds = dbs.getDataDescriptors(clo);
    if (dds.empty()) {
        throw Poco::InvalidArgumentException(
            "There are no data descriptors associated with this "
            "classification object!");
    }


    // Get the process information from the database.
    const int processID = dds.at(0)->processID;
    ProcessPtr process = dbs.getProcess(processID);
    assert (process != nullptr);

    // Get some information about the window-function, window-size and
    // overlap that were used during the creation of the related process.
    // Furthermore we need to know the source sample frequency.
    const double overlap = process->overlap();
    const int windowSize = process->windowSize();
    const WindowFunction winFun = process->windowFunction();
    // Check that all parameters are reasonable.
    if (!winFun || windowSize <= 0 || overlap < 0.0 || overlap >= 1.0) {
        throw Poco::RuntimeException(
            "Unable to retrieve the used window-function, window-size "
            "and/or overlap from the associated process's parameters.");
    }

    // Since we don't know about the dimensionality of the spectrogram yet,
    // we have to use pointers.
    Poco::SharedPtr<Matrix> amplitudeMatrix;
    Poco::SharedPtr<Matrix> phaseMatrix;
    
    if (clo->type == ClassificationObject::Spectrogram) {
        for (vector<DataDescriptorPtr>::const_iterator ddIt = dds.begin();
            ddIt != dds.end(); ++ddIt)
        {
            if (!(*ddIt)->available) {
                throw Poco::InvalidArgumentException(
                    "At least one of this classification object's data "
                    "descriptors is marked as being unavailable!");
            }
            switch ((*ddIt)->type) {
                case DataDescriptor::PhaseMatrix:
                    phaseMatrix = new Matrix(
                        sts.getLocation(*ddIt).toString());
                    break;
                case DataDescriptor::MagnitudeMatrix:
                    amplitudeMatrix = new Matrix(
                        sts.getLocation(*ddIt).toString());
                    break;
                default:
                    BasicApplication::instance().logger()
                        .warning("Unknown data descriptor!");
                    break;
            }
        } 
        if (!amplitudeMatrix.get() || !phaseMatrix.get()) {
            throw Poco::InvalidArgumentException(
                "Either the amplitude- or phase-matrix "
                "are missing for this classification object!");
            return (AudioData*) false;
        }
    }

    // NMFComponent | NMDComponent

    else if (clo->type == ClassificationObject::NMDComponent)
    {
        // Iterate over all data descriptors in order to get the spectrum- and
        // gains-vectors as well as the phase-matrix.
        Poco::SharedPtr<RowVector> gains;
        Poco::SharedPtr<Matrix> spectrum;
        for (vector<DataDescriptorPtr>::const_iterator ddIt = dds.begin(); 
            ddIt != dds.end(); ++ddIt)
        {
            if (!(*ddIt)->available) {
                throw Poco::InvalidArgumentException(
                    "At least one of this classification object's data "
                    "descriptors is marked as being unavailable!");
            }
            switch ((*ddIt)->type) {
            case DataDescriptor::Spectrum:
                spectrum = new Matrix(sts.getLocation((*ddIt)).toString());
                break;
            case DataDescriptor::Gains:
                gains = new RowVector(sts.getLocation((*ddIt)).toString());
                break;
            case DataDescriptor::PhaseMatrix:
                phaseMatrix = new Matrix(sts.getLocation((*ddIt)).toString());
                break;
            case DataDescriptor::MagnitudeMatrix:
                BasicApplication::instance().logger()
                                            .warning("Unknown data descriptor!");
                break;
            case DataDescriptor::FeatureMatrix:
                BasicApplication::instance().logger()
                                            .warning("Unknown data descriptor!");
                break;
            case DataDescriptor::Invalid:
                BasicApplication::instance().logger()
                                            .warning("Unknown data descriptor!");
                break;
            default:
                BasicApplication::instance().logger()
                                            .warning("Unknown data descriptor!");
                break;
            }
        }

        // Check that all neccessary data are available.
        if (!spectrum.get() || !gains.get() || !phaseMatrix.get()) {
            throw Poco::InvalidArgumentException(
                "Either the spectrum, the gains vector or the phase matrix "
                "are missing for this classification object!");
        }

        // Compute the amplitude matrix.
        amplitudeMatrix = new Matrix(spectrum->rows(), gains->dim());
        amplitudeMatrix->zero();
        RowVector componentGains = *gains;
        for (unsigned int t = 0; t < spectrum->cols(); ++t) {
            amplitudeMatrix->add(spectrum->nthColumn(t) * componentGains);
            componentGains.shiftRight();
        }
    }

    // Revert any transformations, in reverse order.
    vector<MatrixTransform*> tfs = process->transforms();
    vector<MatrixTransform*>::const_reverse_iterator tflast(tfs.end());
    vector<MatrixTransform*>::const_reverse_iterator tffirst(tfs.begin());
    for (vector<MatrixTransform*>::const_reverse_iterator rit = tflast;
         rit != tffirst; ++rit) // YES, it's operator ++ ;-)
    {
        // Let Poco::SharedPtr do the dirty work of pointer handling!
        amplitudeMatrix = (*rit)->inverseTransform(amplitudeMatrix);
    }

    AudioData* ad = 0;
    BasicApplication::lockFFTW();
    try {
        ad = AudioData::fromSpectrogram(*amplitudeMatrix, 
                                        *phaseMatrix,
                                        winFun, 
                                        windowSize, 
                                        overlap,
                                        process->sampleFreq, 
                                        obs);
    }
    catch (...) {
        BasicApplication::unlockFFTW();
        throw;
    }
    BasicApplication::unlockFFTW();

    return ad;
}


} // namespace blissart

