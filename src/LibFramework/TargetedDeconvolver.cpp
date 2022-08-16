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


#include <blissart/TargetedDeconvolver.h>

#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>

#include <blissart/DatabaseSubsystem.h>
#include <blissart/BasicApplication.h>
#include <blissart/StorageSubsystem.h>

#include <blissart/BinaryReader.h>

#include <Poco/NumberFormatter.h>

#include <fstream>


using namespace std;
using namespace blissart::linalg;


namespace blissart {


TargetedDeconvolver::TargetedDeconvolver(Matrix& v, unsigned int r,
    const vector<ClassificationObjectPtr>& clObjs,
    Matrix::GeneratorFunction wGenerator,
    Matrix::GeneratorFunction hGenerator) :
    nmf::Deconvolver(v, r, (unsigned int) getNrOfSpectra(*clObjs.begin()),
                     wGenerator, hGenerator)
{
    cout << "\nDecon 1\n";
    buildW(clObjs);
}


TargetedDeconvolver::TargetedDeconvolver(Matrix& v, unsigned int r,
    const vector<string>& matrices,
    Matrix::GeneratorFunction wGenerator,
    Matrix::GeneratorFunction hGenerator,
    bool keepConstant) :
    nmf::Deconvolver(v, r, getNrOfSpectra(*matrices.begin()),
                     wGenerator, hGenerator)
{
    cout << "\nDecon 2\n";
    unsigned int nInitializedCols = (unsigned int) buildW(matrices);
    if (nInitializedCols == r) {
        BasicApplication::instance().logger().
            debug("Keeping all spectra constant.");
        keepWConstant(true);
    } 
    else {
        // This is a hack to get the correct number
        // of constant columns.
        for (unsigned int i = 0; i < nInitializedCols; ++i) {
            BasicApplication::instance().logger().
                debug("Keeping spectrum #" + Poco::NumberFormatter::format(i + 1) +
                    " constant.");
            keepWColumnConstant(i, true);
        }
    }
}
    

TargetedDeconvolver::TargetedDeconvolver(Matrix& v, unsigned int r,
    const vector<int>& clObjIDs,
    Matrix::GeneratorFunction wGenerator,
    Matrix::GeneratorFunction hGenerator) :
    nmf::Deconvolver(v, r, getNrOfSpectra(*clObjIDs.begin()),
                     wGenerator, hGenerator)
{
cout << "\nDecon 3\n";
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
cout << "\nDecon 4\n";
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

        if ((*itr)->type != ClassificationObject::NMDComponent &&
            (*itr)->type != ClassificationObject::Spectrogram) 
        {
            throw Poco::InvalidArgumentException(
                "Invalid classification object type for initialization");
        }

        vector<DataDescriptorPtr> dds = dbs.getDataDescriptors(*itr);
        for (vector<DataDescriptorPtr>::const_iterator dItr = dds.begin();
             dItr != dds.end(); ++dItr)
        {
            if ((*dItr)->type == DataDescriptor::Spectrum ||
                (*dItr)->type == DataDescriptor::MagnitudeMatrix)
            {
                spectrum = 
                    new Matrix(sts.getLocation(*dItr).toString());
            }
        }

        if (spectrum->cols() != _t) {
            throw Poco::InvalidArgumentException(
                "Wrong number of columns in initialization object #"
                + Poco::NumberFormatter::format((*itr)->objectID)
                );
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
TargetedDeconvolver::buildW(const vector<string>& matrices)
{
    unsigned int startComp = 0;
    for (vector<string>::const_iterator itr = matrices.begin(); 
        itr != matrices.end(); ++itr)
    {
        vector<Matrix*> mv = Matrix::arrayFromFile(*itr);
        if (mv.size() != _t) {
            throw Poco::InvalidArgumentException("File " + 
                (*itr) + " contains " + Poco::NumberFormatter::format(mv.size()) + 
                " matrices, but nrOfSpectra = " + Poco::NumberFormatter::format(_t) + "!");
        }
        int cols = 0;
        for (unsigned int t = 0; t < (unsigned int)mv.size(); ++t) {
            // This SharedPtr takes ownership of the pointer.
            // Thus, the current matrix pointer is invalidated at the end of the loop!
            Poco::SharedPtr<Matrix> spectrum(mv[t]);
            //spectrum = new Matrix(*itr);
            BasicApplication::instance().logger().debug(
                "Loaded matrix (" + (*itr) + ") (" 
                + Poco::NumberFormatter::format(spectrum->rows()) + "x" 
                + Poco::NumberFormatter::format(spectrum->cols()) + ")");
            if (startComp + spectrum->cols() > _h.rows()) {
                throw Poco::InvalidArgumentException(
                    "Too many columns in matrix file: " + (*itr));
            }
            if (_w[t]->rows() != spectrum->rows()) {
                throw Poco::InvalidArgumentException(
                    "Wrong dimension of spectrum in matrix file: " + (*itr));
            }
            for (unsigned int j = 0; j < spectrum->cols(); ++j)
            {
                _w[t]->setColumn(startComp + j, spectrum->nthColumn(j));
            }
            if (cols == 0) cols = spectrum->cols();
        }
        startComp += cols;
    }
    BasicApplication::instance().logger().
        debug(Poco::NumberFormatter::format(startComp) +
        " spectra are constant");
    return startComp;
}


int
TargetedDeconvolver::getNrOfSpectra(int clObjID)
{
    cout << "\ngetNrOfSpectra 1\n";
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
    cout << "\ngetNrOfSpectra 2\n";
    DatabaseSubsystem& dbs = 
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    vector<DataDescriptorPtr> dds = dbs.getDataDescriptors(clObj);

    int result = 0;
    if (clObj->type == ClassificationObject::NMDComponent) {
        ProcessPtr process = dbs.getProcess(clObj);
        result = process->spectra();
    }
    else if (clObj->type == ClassificationObject::Spectrogram) {
        for (vector<DataDescriptorPtr>::const_iterator dItr = dds.begin();
             dItr != dds.end(); ++dItr)
        {
            if ((*dItr)->type == DataDescriptor::MagnitudeMatrix)
            {
                StorageSubsystem& sts = BasicApplication::instance().
                    getSubsystem<StorageSubsystem>();
                Matrix tmp (sts.getLocation(*dItr).toString());
                result = tmp.cols();
                break;
            }
        }
    }
    else {
        throw Poco::InvalidArgumentException(
            "Invalid classification object type for initialization");
    }
    return result;
}


int
TargetedDeconvolver::getNrOfSpectra(const std::string &file)
{
    uint32_t nS;
    std::ifstream fis(file.c_str(), std::ios::in | std::ios::binary);
    if (fis.fail())
        return 0;
    BinaryReader br(fis, BinaryReader::LittleEndian);
    uint32_t flag;
    br >> flag;
    if (flag == 2) {
        nS = 1;
    }
    else if (flag == 3) {
        br >> nS;
    }
    else {
        throw Poco::InvalidArgumentException(
            "Not a matrix file: " + file);
    }
    return nS;
}


} // namespace blissart
