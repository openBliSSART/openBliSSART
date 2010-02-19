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


#ifndef __BLISSART_NMF_DECONVOLVER_H__
#define __BLISSART_NMF_DECONVOLVER_H__


#include <common.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/nmf/randomGenerator.h>
#include <cassert>


namespace blissart {


// Forward declaration
class ProgressObserver;


namespace nmf {


/**
 * Performs non-negative matrix deconvolution minimizing either extended
 * KL divergence (Smaragdis 2004) or Euclidean distance (Wang 2009).
 */
class LibNMF_API Deconvolver
{
public:
    /**
     * An enumeration of cost functions which can be used for NMF/NMD.
     */
    typedef enum
    {
        EuclideanDistance,
        KLDivergence,
        EuclideanDistanceSparse,
        KLDivergenceSparse,
        EuclideanDistanceSparseNormalized
    } NMFCostFunction;


    /**
     * Returns a textual description for the given element of the 
     * NMFCostFunction enumeration.
     */
    static const char* costFunctionName(NMFCostFunction cf);


    /**
     * Constructs a Deconvolver object for the matrix V.
     * Initializes the matrix factors with random matrices.
     * @param   v    a Matrix object
     * @param   r    the dimensionality of the factorization
     * @param   t    the desired number of W matrices
     */
    Deconvolver(const blissart::linalg::Matrix& v, unsigned int r, 
        unsigned int t,
        blissart::linalg::Matrix::GeneratorFunction wGenerator 
        = gaussianRandomGenerator,
        blissart::linalg::Matrix::GeneratorFunction hGenerator 
        = gaussianRandomGenerator);

    /**
     * Destroys the Deconvolver object and frees all space used for matrices.
     */
    virtual ~Deconvolver();

    /**
     * Returns the number of components.
     */
    inline unsigned int nrOfComponents() const;

    /**
     * Returns the number of spectra.
     */
    inline unsigned int nrOfSpectra() const;

    /**
     * Returns the W matrix with the given index.
     */
    inline const blissart::linalg::Matrix& getW(unsigned int i) const;

    /**
     * Sets the W matrix with the given index.
     */
    void setW(unsigned int i, const blissart::linalg::Matrix& w);

    /**
     * Sets a flag that controls whether all W matrices are kept constant
     * during the iteration process.
     */
    inline void keepWConstant(bool flag);

    /**
     * Tells the Deconvolver to keep the specified column constant in all W 
     * matrices.
     */
    inline void keepWColumnConstant(unsigned int index, bool flag);

    /**
     * Sets whether the W and H matrices should be normalized.
     */
    inline void setNormalizeMatrices(bool flag);

    /**
     * Returns whether the W and H matrices are normalized.
     */
    inline bool getNormalizeMatrices() const;

    /**
     * Generates all W matrices using the specified generator function.
     */
    void generateW(blissart::linalg::Matrix::GeneratorFunction generator
        = gaussianRandomGenerator);

    /**
     * Returns the H matrix.
     */
    inline const blissart::linalg::Matrix& getH() const;

    /**
     * Sets the H matrix.
     */
    void setH(const blissart::linalg::Matrix& h);

    /**
     * Randomizes the H matrix.
     */
    void generateH(blissart::linalg::Matrix::GeneratorFunction generator
        = gaussianRandomGenerator);

    /**
     * Sets the sparsity parameters for each entry of H, given as a matrix.
     */
    inline void setS(const blissart::linalg::Matrix& s);

    /**
     * Returns the sparsity parameters for each entry of H as a matrix.
     */
    inline const blissart::linalg::Matrix& getS() const;

    /**
     * Returns the current value of approximation of V.
     */
    inline const blissart::linalg::Matrix& getApprox() const;
    
    /**
     * Recomputes the approximation of V, storing it internally.
     * The value can be retrieved using getApprox().
     */
    void computeApprox();

    /**
     * Performs decomposition according to the given cost function.
     * The appropriate algorithm is chosen automatically.
     * @param NMFCostFunction   the cost function to minimize
     * @param maxSteps          maximum number of iteration steps
     * @param eps               if eps > 0.0, convergence of the approximation
     *                          is checked and once it is reached, iteration 
     *                          is stopped; setting eps > 0.0 may significantly
     *                          slow down computation and should not be used
     *                          in production code
     * @param observer          a ProgressObserver that is notified every 25
     *                          iteration steps
     */
    void decompose(NMFCostFunction cf, unsigned int maxSteps, double eps,
                   ProgressObserver *observer = 0);

    /**
     * Returns the absolute error achieved in the last iteration,
     * measured using the Frobenius matrix norm.
     */
    inline double absoluteError();

    /**
     * Returns the relative error achieved in the last iteration,
     * measured using the Frobenius matrix norm.
     */
    inline double relativeError();

    /**
     * Returns the number of steps performed in the last iteration.
     */
    inline unsigned int numSteps() const;


    /**
     * Sets the number of iteration steps after which progress should
     * be reported in the factorization methods (defaults to 25).
     * Small values may result in performance loss while larger values
     * may prevent any progress being seen when factorizing large matrices.
     */
    inline void setProgressNotificationDelay(unsigned int nSteps);


protected:
    /**
     * Performs NMD using a generalized Kulback-Leibler divergence,
     * using Smaragdis' algorithm (2004).
     * Setting eps > 0 significantly slows down the NMD and should therefore 
     * not be used in production code.
     */
    void factorizeNMDKL(unsigned int maxSteps, double eps,
                        ProgressObserver *observer = 0);

    /** 
     * Performs NMD minimizing squared Euclidean distance,
     * according to Wang 2009.
     * Setting eps > 0 significantly slows down the NMD and should therefore 
     * not be used in production code.
     */
    void factorizeNMDED(unsigned int maxSteps, double eps,
                        ProgressObserver *observer = 0);

    // A more efficient implementation of NMD-ED for 1 spectrum (NMF case).
    void factorizeNMFED(unsigned int maxSteps, double eps,
                        ProgressObserver *observer = 0);

    // Sparse NMF according to Virtanen 2007, modified to use Euclidean
    // distance for measurement of reconstruction error.
    void factorizeNMFEDSparse(unsigned int maxSteps, double eps,
                              ProgressObserver *observer = 0);

    // The W update (common for factorizeNMFED and factorizeEDSparse).
    void factorizeNMFEDWUpdate(blissart::linalg::Matrix& w) const;

    // Calculate the update matrices (numerator/denominator) for the H update
    // (common for factorizeNMFED and factorizeNMFEDSparse).
    // Note that the actual update differs between sparse and non-sparse
    // versions.
    void calculateNMFEDHUpdate(blissart::linalg::Matrix& num,
                               blissart::linalg::Matrix& denom) const;

    // Sparse NMF according to Virtanen 2007, measuring reconstruction error
    // using extended KL divergence.
    void factorizeNMFKLSparse(unsigned int maxSteps, double eps,
                              ProgressObserver *observer = 0);

    // Sparse NMF minimizing Euclidean distance, measured using normalized
    // basis vectors (Eggert and Körner 2004).
    void factorizeNMFEDSparseNorm(unsigned int maxSteps, double eps,
                                  ProgressObserver *observer = 0);

    // Checks convergence of the approximation. If recomputeApprox is set,
    // the new approximation is computed first (maybe in the iteration itself
    // the approximation is not needed!)
    bool checkConvergence(double eps, bool recomputeApprox);

    // Increments the internal iteration step counter.
    // Calls the given ProgressObserver, respecting the notification delay.
    // Does nothing if the argument is null.
    void nextItStep(ProgressObserver* observer, unsigned int maxSteps);

    // Computes error of current approximation.
    void computeError();

    // Helper function that efficiently computes the product of W[p] and H,
    // where H is shifted p spots to the right.
    void computeWpH(unsigned int p, blissart::linalg::Matrix& target);

    // Helper function that sets the negative elements of a matrix
    // to a small positive value.
    void ensureNonnegativity(blissart::linalg::Matrix& m, 
        double epsilon = 1e-9);

    // Normalizes W[p] columns to unity for all p
    void normalizeMatrices();

    const blissart::linalg::Matrix& _v;
    blissart::linalg::Matrix        _approx;
    blissart::linalg::Matrix*       _oldApprox;
    blissart::linalg::Matrix**      _w;
    bool                        _wConstant;
    bool*                       _wColConstant;
    bool                        _normalizeMatrices;
    unsigned int                _t;
    blissart::linalg::Matrix        _h;
    blissart::linalg::Matrix        _s;
    unsigned int                _numSteps;
    double                      _absoluteError;
    double                      _relativeError;
    const double                _vFrob;
    unsigned int                _notificationDelay;


private:
    // Forbid copy constructor and operator=.
    Deconvolver(const Deconvolver&);
    Deconvolver& operator=(const Deconvolver&);
};


// Inlines


unsigned int Deconvolver::nrOfComponents() const
{
    return _h.rows();
}


unsigned int Deconvolver::nrOfSpectra() const
{
    return _t;
}


const blissart::linalg::Matrix& Deconvolver::getW(unsigned int i) const
{
    return *(_w[i]);
}


void Deconvolver::keepWConstant(bool flag)
{
    _wConstant = flag;
}


void Deconvolver::keepWColumnConstant(unsigned int index, bool flag)
{
    _wColConstant[index] = flag;
}


void Deconvolver::setNormalizeMatrices(bool flag)
{
    _normalizeMatrices = flag;
}


bool Deconvolver::getNormalizeMatrices() const
{
    return _normalizeMatrices;
}


const blissart::linalg::Matrix& Deconvolver::getH() const
{
    return _h;
}


void Deconvolver::setS(const blissart::linalg::Matrix& s)
{
    assert(_s.rows() == _h.rows() && _s.cols() && _h.cols());
    _s = s;
}


const blissart::linalg::Matrix& Deconvolver::getS() const
{
    return _s;
}


const blissart::linalg::Matrix& Deconvolver::getApprox() const
{
    return _approx;
}


double Deconvolver::absoluteError()
{
    if (_absoluteError < 0)
        computeError();
    return _absoluteError;
}


double Deconvolver::relativeError()
{
    if (_relativeError < 0)
        computeError();
    return _relativeError;
}


unsigned int Deconvolver::numSteps() const
{
    return _numSteps;
}


void Deconvolver::setProgressNotificationDelay(unsigned int numSteps)
{
    _notificationDelay = numSteps;
}


} // namespace nmf

} // namespace blissart


#endif // __BLISSART_NMF_DECONVOLVER_H__
