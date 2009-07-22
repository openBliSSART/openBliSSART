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


#include <blissart/ica/PCA.h>
#include <blissart/linalg/generators/generators.h>
#include <stdexcept>


using namespace blissart::linalg;


namespace blissart {

namespace ica {


PCA::PCA() :
    _meanVector(NULL),
    _covarianceMatrix(NULL),
    _basisChangeMatrix(NULL),
    _pcaDone(false)
{
}


PCA::~PCA()
{
    if (_meanVector) {
        delete _meanVector;
        _meanVector = NULL;
    }
    if (_covarianceMatrix) {
        delete _covarianceMatrix;
        _covarianceMatrix = NULL;
    }
    if (_basisChangeMatrix) {
        delete _basisChangeMatrix;
        _basisChangeMatrix = NULL;
    }
}


PCA* PCA::compute(Matrix* data, unsigned int maxDim, bool isCentered)
{
    // Do a little sanity check on maxDim.
    assert(maxDim <= data->rows());

    // During calculation of the covariance matrix there's a division by
    // (data->cols() - 1), thus we must check that there are at least 2 columns!
    assert(data->cols() > 1);

    // p will encapsulate the results.
    PCA *p = new PCA();

    // Determine the mean vector.
    p->_meanVector = new ColVector(data->meanColumnVector());

    if (!isCentered) {
        // If the mean vectors length doesn't equal zero then the data vectors
        // have to be adjusted accordingly.
        if (p->_meanVector->length() > 0) {
            for (unsigned int i = 0; i < data->rows(); i++)
                for (unsigned int j = 0; j < data->cols(); j++)
                    (*data)(i,j) -= p->_meanVector->at(i);
        }
    } else {
        // Assert that the data really _is_ centered.
        assert(p->_meanVector->length() == 0);
    }

    // Determine the covariance matrix.
    p->_covarianceMatrix = new Matrix(1.0 / ((double)data->cols() - 1.0) *
                                      data->multWithTransposedMatrix(*data));

    // Compute the eigenpairs in descending(!) order.
    p->_eigenPairs = p->_covarianceMatrix->eigenPairs(maxDim);

    // Eigenvalues of 0 indicate 0 variance, and since the eigenvalues have been
    // calculated from most- to least-dominant, all eigenvalues following the
    // first 0 will be 0 as well. The associated eigenvectors have zero length
    // and so we remove them.
    for (Matrix::EigenPairs::iterator it = p->_eigenPairs.begin();
        it != p->_eigenPairs.end();)
    {
        if ((*it).first == 0)
            it = p->_eigenPairs.erase(it);
        else
            ++it;
    }
    // If there are no eigenvalues left, throw a runtime_exception.
    if (p->_eigenPairs.empty()) {
        delete p;
        throw std::runtime_error("The source-matrix contains no information.");
    }

    // Build a transformation matrix whose columns are equal to the computed
    // eigenvectors, thus building a new basis.
    p->_basisChangeMatrix = new Matrix(p->_covarianceMatrix->rows(),
                                       p->_eigenPairs.size());
    for (unsigned int j = 0; j < p->_eigenPairs.size(); j++)
        p->_basisChangeMatrix->setColumn(j, p->_eigenPairs.at(j).second);

    // Multiply the transposed(!) transformation matrix and the data matrix.
    *data = p->_basisChangeMatrix->transposed() * (*data);

    // Indicate that PCA has been done and all corresponding variables may be
    // accessed now.
    p->_pcaDone = true;

    // Return the encapsulated results.
    return p;
}


} // namespace pca

} // namespace blissart
