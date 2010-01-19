//
// $Id: NMDGainsTask.cpp 897 2009-07-07 13:19:25Z felix $
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


#include "NMDGainsTask.h"
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/feature/mfcc.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/HTKWriter.h>
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <Poco/Path.h>


using namespace std;
using namespace blissart;
using blissart::linalg::Matrix;


NMDGainsTask::NMDGainsTask(const string& fileName,
                           int nrOfComponents, int maxIterations,
                           const vector<ClassificationObjectPtr>& initObj,
                           bool allComponents,
                           TransformationMethod method) :
FTTask("NMDGainsTask", fileName),
_initObjects(initObj),
_deconvolver(0),
_nrOfComponents(nrOfComponents),
_maxIterations(maxIterations),
_allComponents(allComponents),
_gainsMatrix(0),
_transformation(method),
_myUniqueID(retrieveUniqueID())
{
}


NMDGainsTask::~NMDGainsTask()
{
    if (_gainsMatrix) {
        delete _gainsMatrix;
        _gainsMatrix = 0;
    }
    if (_deconvolver) {
        delete _deconvolver;
        _deconvolver = 0;
    }
}


void NMDGainsTask::runTask()
{
    incMaxProgress(4.0f);

    // Register a sub-task for the separation whose progress will be constantly updated
    // during NMD iteration.
    registerTask(_myUniqueID, 1);

    // From now on, perform periodical checks to see if the task has been
    // cancelled.
    do {
        readAudioFile();
        incTotalProgress(1.0f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Compute the spectrogram.
        computeSpectrogram();
        incTotalProgress(1.0f);

        // Discard the phase matrix.
        deletePhaseMatrix();

        // Mandatory check.
        if (isCancelled())
            break;

        // Raise hell ;-)
        performNMD();
        setTaskProgress(_myUniqueID, 1, 1.0f);

        // Mandatory check.
        if (isCancelled())
            break;

        // Mandatory check.
        if (isCancelled())
            break;

        // Store / export activation matrices. 
        // We don't want cancellation during the storage process.
        if (_export) {
            exportHTKFile();
        }
        else {
            storeComponents();
        }
        incTotalProgress(1.0f);

        // The original amplitude matrix isn't needed anymore. Free some memory.
        deleteAmplitudeMatrix();

        // Mandatory check.
        if (isCancelled())
            break;
    } while (false);

    // The original amplitude matrix isn't needed anymore. Free some memory.
    // This has to be done again since it is possible that due to user
    // cancellation the upper do-while block was left before the first
    // free-attempt.
    deleteAmplitudeMatrix();
}


void NMDGainsTask::performNMD()
{
    logger().debug(nameAndTaskID() + " initializing deconvolver.");
    if (_nrOfComponents == 0) _nrOfComponents = (int) _initObjects.size();
    _deconvolver = new TargetedDeconvolver(
            amplitudeMatrix(),
            _nrOfComponents,
            _initObjects);

    // All components initialized => keep whole W matrix constant
    if ((unsigned int)_nrOfComponents == _initObjects.size()) {
        _deconvolver->keepWConstant(true);
    }
    // Some components randomized => only keep corresponding W cols constant
    else {
        for (unsigned int c = 0; c < _initObjects.size(); ++c) {
            _deconvolver->keepWColumnConstant(c, true);
        }
    }

    // Raise hell :)
    string cfName = BasicApplication::instance().config().
        getString("blissart.nmdtool.costfunction", "div");
    logger().debug(nameAndTaskID() + " performing NMD.");
    if (cfName.substr(0, 3) == "div") {
        _deconvolver->decompose(TargetedDeconvolver::KLDivergence,
            _maxIterations, 0, this);
    }
    else if (cfName.substr(0, 4) == "dist") {
        _deconvolver->decompose(TargetedDeconvolver::EuclideanDistance,
            _maxIterations, 0, this);
    }
    else {
        throw Poco::InvalidArgumentException("Invalid cost function: " 
            + cfName);
    }
    const Matrix& h = _deconvolver->getH();

    // The number of "relevant" components can be the number of initialized
    // components only or simply the number of NMD components.
    unsigned int relevantComponents = _allComponents ? 
        _nrOfComponents : 
        (unsigned int)_initObjects.size();

    if (_transformation == MaximalIndices) {
        int indexCount = BasicApplication::instance().config().
            getInt("blissart.nmdtool.index_count", 5);
        _gainsMatrix = new Matrix(indexCount, h.cols());
    }
    else {
        _gainsMatrix = new Matrix(relevantComponents, h.cols());
    }

    logger().debug(nameAndTaskID() + " calculating gains matrix.");
    // Transform gains if desired.
    if (_transformation == UnitSum) {
        for (unsigned int j = 0; j < h.cols(); ++j) {
            // Compute length of all gains belonging to "relevant" components.
            double initSum = 0.0;
            for (unsigned int i = 0; i < relevantComponents; ++i) {
                double x = h.at(i, j);
                initSum += x;
            }
            // Normalize gains such that they sum up to 1.
            for (unsigned int i = 0; i < relevantComponents; ++i) {
                _gainsMatrix->setAt(i, j, h.at(i, j) / initSum);
            }

        }
    }
    else if (_transformation == LogDCT) {
        static const double pi = 4.0 * atan(1.0);
        const double normalization = sqrt(2.0 / (double)relevantComponents);
        for (unsigned int j = 0; j < h.cols(); ++j) {
            for (unsigned int i = 0; i < relevantComponents; ++i) {
                double dctCoeff = 0.0;
                for (unsigned int m = 0; m < relevantComponents; ++m) {
                    double x = pi * (double)i / (double)relevantComponents *
                        ((double) m + 0.5);
                    if (h.at(m, j) < 1e-6) {
                        dctCoeff += cos(x) * log(1e-6);
                    }
                    else {
                        dctCoeff += cos(x) * log(h.at(m, j));
                    }
                }
                dctCoeff *= normalization;
                _gainsMatrix->setAt(i, j, dctCoeff);
            }
        }
    }
    else if (_transformation == MaximalIndices) {
        unsigned int indexCount =
            (unsigned int)
                BasicApplication::instance().config().
                getInt("blissart.nmdtool.index_count", 5);
        set<int> indices;
        for (unsigned int j = 0; j < h.cols(); ++j) {
            indices.clear();
            for (unsigned int k = 0; k < indexCount; ++k) {
                double max = 0.0; // H is non-negative
                int maxIndex = 0;
                for (unsigned int i = 0; i < relevantComponents; ++i) {
                    if (indices.find(i) != indices.end()) {
                        continue;
                    }
                    double elem = h.at(i, j);
                    if (elem >= max) {
                        max = elem;
                        maxIndex = i;
                    }
                }
                indices.insert(maxIndex);
                _gainsMatrix->at(k, j) = maxIndex;
            }
        }
    }
    else {
        for (unsigned int j = 0; j < _gainsMatrix->cols(); ++j) {
            for (unsigned int i = 0; i < _gainsMatrix->rows(); ++i) {
                _gainsMatrix->setAt(i, j, h.at(i, j));
            }
        }
    }
}


void NMDGainsTask::storeComponents() const
{
    logger().debug(nameAndTaskID() + " storing feature matrices.");
    DatabaseSubsystem& dbs = BasicApplication::instance().
        getSubsystem<DatabaseSubsystem>();
    StorageSubsystem &sts =  BasicApplication::instance().
        getSubsystem<StorageSubsystem>();

    ProcessPtr process = new Process("NMDGain", fileName(), sampleRate());
    process->setWindowFunction(windowFunction());
    process->setOverlap(overlap());
    process->setWindowSize(windowSize());
    dbs.createProcess(process);

    DataDescriptorPtr ddGains = new DataDescriptor;
    ddGains->type = DataDescriptor::FeatureMatrix;
    ddGains->processID = process->processID;
    dbs.createDataDescriptor(ddGains);
    sts.store(*_gainsMatrix, ddGains);
}


void NMDGainsTask::exportHTKFile() const
{
    Poco::Path inputFile(fileName());
    Poco::Path htkFile = Poco::Path(_exportDir).append(fileName());
    htkFile.setExtension("htk");
    // ofstream needs a char pointer, thus we have to create a string object
    string htkFileString = htkFile.toString();
    ofstream htkFileStream(htkFileString.c_str());
    Poco::Int32 sp = (Poco::Int32)
        (windowSize() * 10e4 * (1.0 - overlap()));
    HTKWriter::writeMatrix(htkFileStream, *_gainsMatrix, sp);
}


void NMDGainsTask::progressChanged(float progress)
{
    setTaskProgress(_myUniqueID, 1, progress);
}
