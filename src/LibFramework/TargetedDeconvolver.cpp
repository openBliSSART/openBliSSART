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


#include <blissart/TargetedDeconvolver.h>

#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>

#include <blissart/DatabaseSubsystem.h>
#include <blissart/BasicApplication.h>
#include <blissart/StorageSubsystem.h>

#include <Poco/NumberFormatter.h>


using namespace std;
using namespace blissart::linalg;


namespace blissart {


TargetedDeconvolver::TargetedDeconvolver(const Matrix& v, unsigned int r,
    const vector<ClassificationObjectPtr>& clObjs,
    Matrix::GeneratorFunction wGenerator,
    Matrix::GeneratorFunction hGenerator) :
    nmf::Deconvolver(v, r, getNrOfSpectra(*clObjs.begin()), 
                     wGenerator, hGenerator)
{
    buildW(clObjs);
}


TargetedDeconvolver::TargetedDeconvolver(const Matrix& v, unsigned int r,
    const vector<int>& clObjIDs,
    Matrix::GeneratorFunction wGenerator,
    Matrix::GeneratorFunction hGenerator) :
    nmf::Deconvolver(v, r, getNrOfSpectra(*clObjIDs.begin()),
                     wGenerator, hGenerator)
{
    DatabaseSubsystem& dbs = 
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    vector<ClassificationObjectPtr> clObjs;
    for (vector<int>::const_iterator itr = clObjIDs.begin(); 
        itr != clObjIDs.end(); ++itr)
    {
        clObjs.push_back(dbs.getClassificationObject(*itr));
    }
    buildW(clObjs);
}


void 
TargetedDeconvolver::buildW(const vector<ClassificationObjectPtr>& clObjs)
{
    DatabaseSubsystem& dbs = 
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    StorageSubsystem& sts = 
        BasicApplication::instance().getSubsystem<StorageSubsystem>();

    unsigned int compIndex = 0;
    for (vector<ClassificationObjectPtr>::const_iterator 
         itr = clObjs.begin(); itr != clObjs.end();
         ++itr, ++compIndex)
    {
        Poco::SharedPtr<Matrix> spectrum;

        if ((*itr)->type != ClassificationObject::NMDComponent) {
            throw Poco::InvalidArgumentException(
                "Invalid classification object type for initialization");
        }

        vector<DataDescriptorPtr> dds = dbs.getDataDescriptors(*itr);
        for (vector<DataDescriptorPtr>::const_iterator dItr = dds.begin();
             dItr != dds.end(); ++dItr)
        {
            if ((*dItr)->type == DataDescriptor::Spectrum ||
                (*dItr)->type == DataDescriptor::MelSpectrum)
            {
                spectrum = 
                    new Matrix(sts.getLocation(*dItr).toString());
            }
        }

        for (unsigned int t = 0; t < spectrum->cols(); ++t) {
            if (_w[t]->rows() != spectrum->rows()) {
                throw Poco::InvalidArgumentException(
                    "Wrong dimension of spectrum in classification object #"
                    + Poco::NumberFormatter::format((*itr)->objectID)
                    );
            }
            _w[t]->setColumn(compIndex, spectrum->nthColumn(t));
        }
    }
}


int
TargetedDeconvolver::getNrOfSpectra(int clObjID)
{
    DatabaseSubsystem& dbs = 
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    ClassificationObjectPtr clObj = dbs.getClassificationObject(clObjID);
    if (clObj.isNull()) {
        throw Poco::NotFoundException("No classification object found with ID "
            + Poco::NumberFormatter::format(clObjID));
    }
    return getNrOfSpectra(clObj);
}


int
TargetedDeconvolver::getNrOfSpectra(ClassificationObjectPtr clObj)
{
    DatabaseSubsystem& dbs = 
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    ProcessPtr process = dbs.getProcess(clObj);
    return process->spectra();
}


} // namespace blissart
