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
                 nmf::Deconvolver::NMDCostFunction cf,
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
    _sparsity = BasicApplication::instance().config().
        getDouble("blissart.separation.activationSparsity.weight", 0.0);
    logger().debug(nameAndTaskID() + ": activation sparsity weight = " 
        + Poco::NumberFormatter::format(_sparsity));
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
    if (numInitializationObjects() > 0 || numInitializationMatrices() > 0) {
        // precedence?
        if (numInitializationMatrices() > 0) {
            _deconvolver =
                new TargetedDeconvolver(
                        amplitudeMatrix(),
                        nrOfComponents(),
                        initializationMatrices(),
                        generatorFunction(), generatorFunction(),
                        constantInitializedComponentSpectra());
        }
        else {
            _deconvolver =
                new TargetedDeconvolver(
                        amplitudeMatrix(),
                        nrOfComponents(),
                        initializationObjects(),
                        generatorFunction(), generatorFunction());
            if (constantInitializedComponentSpectra()) {
                logger().debug("Keeping spectra constant.");
                if (numInitializationObjects() == nrOfComponents()) {
                    _deconvolver->keepWConstant(true);
                } 
                else {
                    for (unsigned int i = 0; i < numInitializationObjects(); ++i) {
                        logger().debug("Keeping spectrum #" + 
                            Poco::NumberFormatter::format(i + 1) +
                            " constant.");
                        _deconvolver->keepWColumnConstant(i, true);
                    }
                }
            }
        }
        if (nrOfSpectra() != _deconvolver->nrOfSpectra()) {
            throw Poco::InvalidArgumentException(
                "Wrong number of spectra in initialization.");
        }
    } 
    else {
        _deconvolver =
            new nmf::Deconvolver(
                    amplitudeMatrix(),
                    nrOfComponents(),
                    nrOfSpectra(),
                    generatorFunction(), generatorFunction());
    }

    _deconvolver->setProgressNotificationDelay(
        BasicApplication::instance().config().
            getInt("blissart.separation.notificationSteps", 25)
    );

    // Currently the sparsity and continuity parameters are the same for the
    // whole H matrix. 
    _deconvolver->setSparsity(nmf::Deconvolver::DefaultSparsityTemplate(_sparsity));

    double wSparsity = BasicApplication::instance().config().
        getDouble("blissart.separation.baseSparsity.weight", 0.0);
    if (wSparsity > 0.0) {
        logger().debug("Setting base sparsity weight to " 
            + Poco::NumberFormatter::format(wSparsity));
        double wSparsityExp = BasicApplication::instance().config().
            getDouble("blissart.separation.baseSparsity.exp", 1.0);
        if (wSparsityExp == 1.0) {
            _deconvolver->setWSparsity(
                nmf::Deconvolver::DefaultSparsityTemplate(wSparsity));
        }
        else {
            logger().debug("Setting base sparsity exponentiation to " 
                + Poco::NumberFormatter::format(wSparsityExp));
            _deconvolver->setWSparsity(
                nmf::Deconvolver::ExponentialSparsityTemplate(wSparsity, wSparsityExp));
        }
    }

    Matrix c(nrOfComponents(), amplitudeMatrix().cols());
    for (unsigned int j = 0; j < c.cols(); ++j) {
        for (unsigned int i = 0; i < c.rows(); ++i) {
            c(i, j) = _continuity;
        }
    }
    _deconvolver->setContinuity(c);
}


void NMDTask::performSeparation()
{
    debug_assert(&amplitudeMatrix() && _deconvolver);

    logger().debug(nameAndTaskID() + " factorizing.");
    nmf::Deconvolver::SparsityConstraint sparsity = nmf::Deconvolver::NoSparsity;
    if (_sparsity > 0.0) {
        string sparsityStr = BasicApplication::instance().config().
            getString("blissart.separation.activationSparsity.cost", "L1Norm");
        // TODO: move this conversion to Deconvolver class
        if (sparsityStr == "L1Norm") {
            logger().debug("Penalizing activations by L1 sparsity constraint.");
            sparsity = nmf::Deconvolver::L1Norm;
        }
        else if (sparsityStr == "NormalizedL1Norm") {
            logger().debug("Penalizing activations by normalized L1 sparsity constraint.");
            sparsity = nmf::Deconvolver::NormalizedL1Norm;
        }
        else {
            logger().warning("Invalid sparsity constraint: " + sparsityStr);
        }
    }

    nmf::Deconvolver::MatrixNormalization norm 
        = nmf::Deconvolver::NormWColumnsEucl;
    string normalizationStr = BasicApplication::instance().config().
        getString("blissart.separation.normalization", "Wcol_L2Norm");
    if (normalizationStr == "Wcol_L2Norm") {
        logger().debug("Normalizing W columns by L2 norm.");
        norm = nmf::Deconvolver::NormWColumnsEucl;
    }
    else if (normalizationStr == "H_L2Norm") {
        logger().debug("Normalizing H by Frobenius norm.");
        norm = nmf::Deconvolver::NormHFrob;
    }
    else if (normalizationStr == "none") {
        norm = nmf::Deconvolver::NoNorm;
    }
    _deconvolver->setNormalization(norm);

    _deconvolver->decompose(_cf, maxIterations(), epsilon(), 
                            sparsity, getContinuity() > 0.0, this);
}


void NMDTask::setProcessParameters(ProcessPtr process) const
{
    SeparationTask::setProcessParameters(process);
    process->parameters["costFunction"] = nmf::Deconvolver::costFunctionName(_cf);
    process->parameters["sparsity"] = Process::formatDouble(_sparsity);
    process->parameters["normalizeMatrices"] = BasicApplication::instance().config().
        getString("blissart.separation.normalization", "Wcol_L2Norm");
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
