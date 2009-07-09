//
// $Id: NMFTask.cpp 886 2009-07-01 14:33:24Z alex $
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


#include <blissart/NMFTask.h>
#include <blissart/BasicApplication.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/nmf/Factorizer.h>

#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/Matrix.h>

#include <Poco/Exception.h>
#include <Poco/NumberFormatter.h>


using namespace std;
using namespace blissart::linalg;


namespace blissart {


NMFTask::NMFTask(const std::string &fileName,
                 SeparationTask::DataKind dataKind,
                 Algorithm algorithm, int nrOfComponents,
                 int maxIterations, double epsilon, bool isVolatile) :
    SeparationTask(
        SeparationTask::NMF, "NMFTask", dataKind,
        fileName, nrOfComponents, 1,
        maxIterations, epsilon, isVolatile
    ),
    _factorizer(0),
    _algorithm(algorithm)
{
    logger().debug(nameAndTaskID() + " initialized.");
}


NMFTask::~NMFTask()
{
    if (_factorizer) {
        delete _factorizer;
        _factorizer = 0;
    }
}


void NMFTask::initialize()
{
    // To avoid duplicate code, we currently provide initialization for NMD
    // only, as NMF is a special case of NMD with T = 1.
    if (numInitializationObjects() > 0) {
        throw Poco::NotImplementedException("NMF cannot be pre-initialized");
    }

    debug_assert(&amplitudeMatrix() && !_factorizer);
    _factorizer = new nmf::Factorizer(amplitudeMatrix(), nrOfComponents());
}


void NMFTask::performSeparation()
{
    debug_assert(&amplitudeMatrix());

    logger().debug(nameAndTaskID() + " factorizing.");

    switch (_algorithm) {
        case GradientDescent:
            _factorizer->factorizeGradient(maxIterations(), epsilon(), this);
            break;
        case MUDistance:
            _factorizer->factorizeDistance(maxIterations(), epsilon(), this);
            break;
        case MUDivergence:
            _factorizer->factorizeDivergence(maxIterations(), epsilon(), this);
            break;
        default:
            throw Poco::RuntimeException("Unknown NMF algorithm.");
    }
}


const Matrix& NMFTask::magnitudeSpectraMatrix(unsigned int index) const
{
    return _factorizer->getFirst();
}


const Matrix& NMFTask::gainsMatrix() const
{
    return _factorizer->getSecond();
}


void NMFTask::progressChanged(float progress)
{
    setSeparationProgress(progress);
}


} // namespace blissart
