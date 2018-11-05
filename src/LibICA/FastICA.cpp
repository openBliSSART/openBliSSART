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


#include <blissart/ica/FastICA.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/generators/generators.h>

#include <cmath>
#include <cassert>
#include <stdexcept>

#include <iostream>
using namespace std;

using namespace blissart::linalg;

namespace blissart {

namespace ica {


FastICA::FastICA(Matrix* data, unsigned int nSources, double prec) :
    _data(data),
    _mixingMatrix(NULL),
    _nSources(nSources),
    _prec(prec),
    _dataCentered(false),
    _whiteningDone(false)
{
}


FastICA::~FastICA()
{
    // _data is NOT owned by FastICA.
    if (_mixingMatrix) {
        delete _mixingMatrix;
        _mixingMatrix = NULL;
    }
}


Matrix FastICA::decorrelationMatrix(const Matrix& C)
{
    debug_assert(_dataCentered);

    // Get the eigenpairs of the covariance matrix.
    Matrix::EigenPairs eigenp = C.eigenPairs(_nSources);

    // Eigenvalues of 0 indicate 0 variance, and since the eigenvalues have been
    // calculated from most- to least-dominant, all eigenvalues following the
    // first 0 will be 0 as well. The associated eigenvectors have zero length
    // and so we remove them.
    for (Matrix::EigenPairs::iterator it = eigenp.begin(); it != eigenp.end();) {
        if (it->first == 0)
            it = eigenp.erase(it);
        else
            ++it;
    }
    // If there are no eigenvalues left, throw a runtime_exception.
    if (eigenp.empty())
        throw std::runtime_error("The source-matrix contains no information.");

    // Assemble two matrices D and E, where D's columns are the reciprocals
    // of the square roots of the eigenvalues (= standard deviation) and E's
    // columns are the respective eigenvectors.
    const unsigned int dim = eigenp.size();
    Matrix E(C.rows(), dim);
    Matrix D(dim, dim, generators::zero);
    for (unsigned int i = 0; i < dim; i++) {
        E.setColumn(i, eigenp.at(i).second);
        D(i,i) = 1.0 / sqrt(eigenp.at(i).first);
    }

    // If eigenvalues of 0 showed up, thus indicating that certain data do not
    // contain any information and hence should be removed, we return D * E^T
    // instead of E * D * E^T. This is acceptable because the remaining steps
    // of the ICA only deal with finding the suitable rotation in order to
    // achieve the desired statistical independence.
    if (C.rows() != dim)
        return D.multWithTransposedMatrix(E);
    else
        return E * D.multWithTransposedMatrix(E);
}


void FastICA::centerData()
{
    debug_assert(!_dataCentered);

    // Get the expected value of the _data matrix.
    const ColVector cv = _data->meanColumnVector();
    // Subtract the expected value from each column of the _data matrix.
    if (cv.length() > 0) {
        for (unsigned int i = 0; i < _data->rows(); i++) {
            double c = cv.at(i);
            for (unsigned int j = 0; j < _data->cols(); j++)
                (*_data)(i,j) -= c;
        }
    }
    _dataCentered = true;
}


void FastICA::performWhitening(bool isCentered)
{
#ifdef _DEBUG
    _data->dumpRowMajor("before_whitening.plt");
#endif

    // Center the data vectors if neccessary.
    if (!isCentered)
        centerData();
    else
        _dataCentered = true;

    // Calculate the covariance matrix.
    Matrix covMatrix = 1.0 / (_data->cols() - 1.0) *
                       _data->multWithTransposedMatrix(*_data);

    // Now whitening can be done by multiplying a suitable decorrelation matrix
    // by the data matrix.
    // Note that after this multiplication the number of rows of the data matrix
    // can be reduced (see \ref decorrelationMatrix).
    *_data = decorrelationMatrix(covMatrix) * (*_data);

    // If the sample matrix has more rows than the number of sources that should
    // be separated, then the matrix' superfluos rows should be removed _now_ in
    // order to reduce noise.
    if (_data->rows() > _nSources)
        *_data = _data->upToAndIncludingRow(_nSources - 1 /* zero indexed */);

#ifdef _DEBUG
    _data->dumpRowMajor("after_whitening.plt");
#endif

    _whiteningDone = true;
}


ColVector FastICA::expValCallBack(const ColVector& cv, void* info)
{
    struct info_t* i = (struct info_t*)info;
    double t = tanh(*(i->wt) * cv);
    return cv * t - (1.0 - t * t) * *(i->w);
}


void FastICA::computeMixingMatrix(const unsigned int maxIterations)
{
    debug_assert(_dataCentered && _whiteningDone);

    ColVector w(_nSources);
    ColVector wn(_nSources);
    RowVector wt(_nSources);
    const ColVector zeroVector(_data->rows(), generators::zero);
    bool errFlag;
    struct info_t info;
    std::vector<ColVector> mixers;

    _nrOfConvergenceErrors = 0;
    for (unsigned int i = 0; i < _nSources; i++) {
        w.randomize();
        errFlag = true;
        for (unsigned int j = 0; j < maxIterations; j++) {
            wt = w.transposed();
            info.w = &w;
            info.wt = &wt;
            wn = _data->expectedValue<blissart::linalg::ColVector>(zeroVector,
                                                 &info,
                                                 expValCallBack);
            wn.normalize();
            if (i > 0) {
                // Decorrelation of the previous estimates.
                const RowVector wnpt = wn.transposed();
                for (unsigned int k = 0; k < i; k++) {
                    const ColVector& wj = mixers.at(k);
                    wn.sub(wnpt * wj * wj);
                }
                wn.normalize();
            }
            if (fabs(fabs(wn.transposed() * w) - 1.0) < _prec) {
                w = wn;
                // Convergence, hence no error.
                errFlag = false;
                break;
            }
            w = wn;
        }
        // In case of an convergence error increase the related variable.
        if (errFlag) _nrOfConvergenceErrors++;

        // Store the computed vector - no matter if converged or not.
        mixers.push_back(w);
    }

    // Allocate the mixing matrix and fill in the calculated "mixers" as column
    // vectors.
    _mixingMatrix = new Matrix(_nSources, _nSources);
    for (unsigned int i = 0; i < _mixingMatrix->cols(); i++)
        _mixingMatrix->setColumn(i, mixers.at(i));
}

FastICA* FastICA::compute(Matrix* data, unsigned int nSources, bool isCentered,
                          const unsigned int maxIterations, const double prec)
{
    // One restriction of ICA is that there must be a sufficient number of
    // datasets:
    assert(data->rows() >= nSources);

    // During calculation of the covariance matrix there is a division by
    // (_data->cols() - 1). This is why we must make sure at this point that
    // there are at least 2 sources:
    if (data->cols() < 2)
        throw std::runtime_error("Insufficient number of sources.");

    // ica will encapsulate the results.
    FastICA *ica = new FastICA(data, nSources, prec);

    // Through whitening we make sure that the data vectors get centered and
    // that we have normalized variances.
    ica->performWhitening(isCentered);

    // Now the neccessary iteration steps can be performed.
    ica->computeMixingMatrix(maxIterations);

    // The final separation step involves the multiplication of the
    // transposed(!) of the computed basis change matrix by the data matrix.
    *data = ica->_mixingMatrix->transposed() * (*data);

#ifdef _DEBUG
    data->dumpRowMajor("after_fastica.plt");
#endif

    // Return the encapsulated results.
    return ica;
}


} // namespace ica

} // namespace blissart
