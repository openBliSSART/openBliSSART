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


#include <blissart/NMDTask.h>
#include <blissart/BasicApplication.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/nmf/Deconvolver.h>
#include <blissart/TargetedDeconvolver.h>

#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/Matrix.h>

#include <Poco/Exception.h>
#include <Poco/NumberFormatter.h>


using namespace std;
using namespace blissart::linalg;


namespace blissart {


NMDTask::NMDTask(const std::string &fileName,
                 nmf::Deconvolver::NMFCostFunction cf,
                 int nrOfComponents, int nrOfSpectra,
                 int maxIterations, double epsilon, bool isVolatile) :
    SeparationTask(
        SeparationTask::NMD, "NMDTask",
        fileName, nrOfComponents, nrOfSpectra,
        maxIterations, epsilon, isVolatile
    ),
    _deconvolver(0),
    _cf(cf)
{
    logger().debug(nameAndTaskID() + " initialized.");
    // XXX: config option for normalization?
}


NMDTask::~NMDTask()
{
    if (_deconvolver) {
        delete _deconvolver;
        _deconvolver = 0;
    }
}


void NMDTask::initialize()
{
    debug_assert(!_deconvolver);

    // Targeted initialization desired?
    if (numInitializationObjects() > 0) {
        _deconvolver =
            new TargetedDeconvolver(
                    amplitudeMatrix(),
                    nrOfComponents(),
                    initializationObjects(),
                    generatorFunction(), generatorFunction());
        if (nrOfSpectra() != _deconvolver->nrOfSpectra()) {
            throw Poco::InvalidArgumentException(
                "Wrong number of spectra in initialization.");
        }
        if (constantInitializedComponentSpectra()) {
            if (numInitializationObjects() == nrOfComponents()) {
                _deconvolver->keepWConstant(
                        constantInitializedComponentSpectra()
                );
            } else {
                for (unsigned int i = 0; i < numInitializationObjects(); ++i)
                    _deconvolver->keepWColumnConstant(i, 
                        constantInitializedComponentSpectra());
            }
        }
    } else {
        _deconvolver =
            new nmf::Deconvolver(
                    amplitudeMatrix(),
                    nrOfComponents(),
                    nrOfSpectra(),
                    generatorFunction(), generatorFunction());
        _deconvolver->setProgressNotificationDelay(
            BasicApplication::instance().config().
                getInt("blissart.separation.notificationSteps", 25)
        );
    }

    // Currently the sparsity and continuity parameters are the same for the
    // whole H matrix, but the Deconvolver allows a different parameter for
    // each entry.

    Matrix s(nrOfComponents(), amplitudeMatrix().cols());
    for (unsigned int j = 0; j < s.cols(); ++j) {
        for (unsigned int i = 0; i < s.rows(); ++i) {
            s(i, j) = _sparsity;
        }
    }
    _deconvolver->setSparsity(s);

    Matrix c(nrOfComponents(), amplitudeMatrix().cols());
    for (unsigned int j = 0; j < c.cols(); ++j) {
        for (unsigned int i = 0; i < c.rows(); ++i) {
            c(i, j) = _continuity;
        }
    }
    _deconvolver->setContinuity(c);

    _deconvolver->setNormalizeMatrices(_normalizeMatrices);
}


void NMDTask::performSeparation()
{
    debug_assert(&amplitudeMatrix() && _deconvolver);

    logger().debug(nameAndTaskID() + " factorizing.");
    _deconvolver->decompose(_cf, maxIterations(), epsilon(), this);
}


void NMDTask::setProcessParameters(ProcessPtr process) const
{
    SeparationTask::setProcessParameters(process);
    process->parameters["costFunction"] = nmf::Deconvolver::costFunctionName(_cf);
    process->parameters["sparsity"] = Process::formatDouble(_sparsity);
}


const Matrix& NMDTask::magnitudeSpectraMatrix(unsigned int index) const
{
    return _deconvolver->getW(index);
}


const Matrix& NMDTask::gainsMatrix() const
{
    return _deconvolver->getH();
}


void NMDTask::progressChanged(float progress)
{
    setSeparationProgress(progress);
}


void NMDTask::computeRelativeError()
{
    if (_deconvolver != 0) 
        _relativeError = _deconvolver->relativeError();
}


} // namespace blissart
