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

#ifndef __BLISSART_ICA_FASTICA_H__
#define __BLISSART_ICA_FASTICA_H__


#include <common.h>


namespace blissart {


// Forward declaration
namespace linalg {
    class Matrix;
    class ColVector;
    class RowVector;
}


/**
 * Classes related to Independent Component Analysis.
 */
namespace ica {


/**
 * \defgroup ica Independent Component Analysis (LibICA)
 * \addtogroup ica
 * @{
 */


/**
 * Performs Independent Component Analysis using the FastICA algorithm by
 * Hyvaerinen.
 */
class LibICA_API FastICA
{
public:
    /**
     * Destructs an instance of FastICA.
     */
    virtual ~FastICA();


    /**
     * Performs ICA on the given data matrix for the given parameters.
     * @param   data            a pointer to the data matrix whose columns are
     *                          the sample
     * @param   nSources        the # of sources that should be separated
     * @param   isCentered      used to indicate whether the data is already
     *                          centered (and thus saving computation time)
     * @param   maxIterations   the maximum # of iterations during FastICA
     * @param   prec            the desired precision
     * @throw                   std::runtime_error
     * @return                  a pointer to a FastICA object
     */
    static FastICA* compute(linalg::Matrix* data, unsigned int nSources,
                            bool isCentered = false,
                            const unsigned int maxIterations = 20,
                            const double prec = 1e-10);


    /**
     * Get the # of convergence errors that occured during FastICA.
     * @return                  the # of convergence errors
     */
    inline unsigned int nrOfConvergenceErrors() const;


protected:
    /**
     * Construct a FastICA object for the given arguments.
     * @param   data            a pointer to a matrix whose columns are the
     *                          sample probes
     * @param   nSources        how many source should be separated
     * @param   prec            the desired precision
     */
    FastICA(linalg::Matrix* data, unsigned int nSources, double prec);


    /**
     * Computes a decorrelation matrix  based on eigenvalue-decomposition and
     * normalization of the given covariance matrix.
     * \attention The returned matrix can be \e rectangular instead of
     * \e quadratic. This is the case iff one or more eigenvalues of 0 showed
     * up, thus indicating that certain data do not contain any information and
     * hence will be removed when multiplying the returned matrix by the
     * data matrix that corresponds to the given covariance matrix.
     * @param   C               the covariance matrix of the centered data
     * @throw                   std::runtime_error
     * @return                  a decorrelation matrix
     */
    linalg::Matrix decorrelationMatrix(const linalg::Matrix& C);


    /**
     * Centers all data, i.e. for each column of the data matrix subtracts the
     * expected value.
     */
    void centerData();


    /**
     * Perform whitening through centering and eigenvalue-decomposition.
     * @param  isCentered       indicates that the data have already been
     *                          centered
     */
    void performWhitening(bool isCentered);


    /**
     * Compute the basis change matrix by means of performing the neccessary
     * FastICA iterations.
     * @param   maxIterations   the maximum # of iterations
     */
    void computeMixingMatrix(const unsigned int maxIterations);


private:
    /*
     * Forbid copy constructor and operator=. Such methods could of course
     * create a copy of the mixing matrix, but since we're dealing with
     * potentially huge matrices, it's probably better to simply forbid these
     * methods.
     */
    FastICA(const FastICA&);
    FastICA& operator = (const FastICA&);


    /**
     * Nested helper struct for the expected value computations during FastICA
     * iteration.
     */
    struct info_t {
        const linalg::ColVector* w;
        const linalg::RowVector* wt;
    };


    /**
     * Static helper callback function for the neccessary expected value
     * computations during FastICA iteration.
     */
    static linalg::ColVector expValCallBack(const linalg::ColVector& cv,
                                            void* info);


    linalg::Matrix*     _data;
    linalg::Matrix*     _mixingMatrix;
    const unsigned int  _nSources;
    const double        _prec;
    unsigned int        _nrOfConvergenceErrors;
    bool                _dataCentered;
    bool                _whiteningDone;
};


/**
 * @}
 */


// Inlines


inline unsigned int FastICA::nrOfConvergenceErrors() const
{
    return _nrOfConvergenceErrors;
}


} // namespace ica

} // namespace blissart


#endif // __BLISSART_ICA_FASTICA_H__
