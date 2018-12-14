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


#ifndef __BLISSART_ICA_PCA_H__
#define __BLISSART_ICA_PCA_H__


#include <common.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>

#include <cassert>
#include <vector>


namespace blissart {

namespace ica {


/**
 * \addtogroup ica
 * @{
 */

/**
 * Performs Principal Components Analysis.
 */
class LibICA_API PCA
{
public:
    /**
     * Destructs an instance of PCA.
     */
    virtual ~PCA();


    /**
     * Perform PCA on the given data matrix.
     * @param   data        a pointer to a matrix with sample probes as columns
     * @param   maxDim      the maximum dimension after PCA
     * @param   isCentered  indicates whether centering has already been done
     *                      (used to to speed up computations)
     * @throw               std::runtime_error
     * @return              a pointer to a PCA object holding information about
     *                      the basis change matrix, mean vector, etc.
     */
    static PCA* compute(blissart::linalg::Matrix* data, unsigned int maxDim,
                        bool isCentered = false);


    /**
     * Returns the expected value of the source data.
     * @return              the expected value as ColVector
     */
    inline const linalg::ColVector& expectedValue() const;


    /**
     * Returns the covariance matrix.
     * @return              a pointer to the covariance matrix
     */
    inline const linalg::Matrix& covarianceMatrix() const;


    /**
     * Returns the eigenpairs of the covariance matrix.
     * @return              the eigenpairs of the covariance matrix
     */
    inline const linalg::Matrix::EigenPairs& eigenPairs() const;


    /**
     * Returns the basis change matrix that was computed during PCA.
     * @return              a pointer to the basis change matrix
     */
    inline const linalg::Matrix& basisChangeMatrix() const;


protected:
    /**
     * Constructs a PCA object.
     */
    PCA();


private:
    // Forbid copy constructor and operator=. Such methods could of course
    // create a copy of the encapsulated matrices, but since we're dealing with
    // potentially huge matrices, it's probably better to simply forbid these
    // methods.
    PCA(const PCA&);
    PCA& operator=(const PCA&);


    linalg::ColVector* _meanVector;
    linalg::Matrix*    _covarianceMatrix;
    linalg::Matrix*    _basisChangeMatrix;
    bool               _pcaDone;
#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable:4251)
#endif
    linalg::Matrix::EigenPairs _eigenPairs;
#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(pop)
#endif
};


/**
 * @}
 */


// Inlines


const linalg::ColVector& PCA::expectedValue() const
{
    assert(_pcaDone);
    return *_meanVector;
}


const linalg::Matrix& PCA::covarianceMatrix() const
{
    assert(_pcaDone);
    return *_covarianceMatrix;
}


const linalg::Matrix::EigenPairs& PCA::eigenPairs() const
{
    assert(_pcaDone);
    return _eigenPairs;
}


} // namespace ica

} // namespace blissart

#endif // __BLISSART_ICA_PCA_H__
