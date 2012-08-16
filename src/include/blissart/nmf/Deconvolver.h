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
#include <config.h>
#ifdef HAVE_CUDA
#include <blissart/linalg/GPUMatrix.h>
typedef blissart::linalg::GPUMatrix Matrix_t;
#else
typedef blissart::linalg::Matrix Matrix_t;
#endif
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
     * Represents a template to generate a sparsity matrix.
     */
    class SparsityTemplate
    {
    public:
        virtual double sparsity(int p, int i, int j) const = 0;
    };

    class DefaultSparsityTemplate: public SparsityTemplate
    {
    public:
        DefaultSparsityTemplate(double sparsity) : _sparsity(sparsity) { }
        virtual double sparsity(int p, int i, int j) const 
        { 
            return _sparsity; 
        }

    private:
        double _sparsity;
    };

    /**
     * Sparsity template with exponential increase depending on the time
     * index p.
     */
    class ExponentialSparsityTemplate:  public SparsityTemplate
    {
    public:
        ExponentialSparsityTemplate(double sparsity, double decay) :
            _sparsity(sparsity), _decay(decay) {}
        virtual double sparsity(int p, int i, int j) const;
    private:
        double _sparsity;
        double _decay;
    };
    
    /**
     * Sparsity template where the sparsity depends on the column.
     * E.g., to use different W sparsity values for different components.
     */
    class ColumnSparsityTemplate: public SparsityTemplate
    {
        // TODO
    };

    /**
     * Sparsity template where the sparsity depends on the row.
     * E.g., to use different H sparsity values for different components.
     */
    class RowSparsityTemplate: public SparsityTemplate
    {
        // TODO
    };

    /**
     * An enumeration of cost functions which can be used for NMF/NMD.
     */
    typedef enum
    {
        EuclideanDistance,
        KLDivergence,
        ISDivergence,
        BetaDivergence,
        NormalizedEuclideanDistance
    } NMDCostFunction;


    /**
     * The algorithm to use.
     */
    typedef enum
    {
        Auto,
        NMFEDIncomplete,
        Overcomplete
    } NMDAlgorithm;


    /**
     * Represents different sparsity constraints.
     */
    typedef enum
    {
        NoSparsity,
        L1Norm,
        NormalizedL1Norm
    } SparsityConstraint;
    
    
    /**
     * Various methods to normalize NMF matrices after computation.
     */
    typedef enum {
        NoNorm,           // no normalization
        NormHFrob,        // normalize H to unity Frobenius norm
        NormWColumnsEucl  // normalize each column of W to unity Euclidean norm
    } MatrixNormalization;


    /**
     * Returns a textual description for the given element of the 
     * NMDCostFunction enumeration.
     */
    static const char* costFunctionName(NMDCostFunction cf);


    /**
     * Constructs a Deconvolver object for the matrix V.
     * Initializes the matrix factors with random matrices.
     * @param   v    a Matrix object
     * @param   r    the dimensionality of the factorization
     * @param   t    the desired number of W matrices
     * @param   wGenerator   a GeneratorFunction for the W matrices
     * @param   hGenerator   a GeneratorFunction for the H matrix
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
     * TODO
     */
    inline void setAlgorithm(NMDAlgorithm a);

    /**
     * TODO
     */
    inline NMDAlgorithm getAlgorithm() const;

    /**
     * Controls the matrix normalization that is applied after each update 
     * iteration.
     */
    inline void setNormalization(MatrixNormalization norm);

    /**
     * Returns the matrix normalization method that is used after each update
     * iteration.
     */
    inline MatrixNormalization getNormalization() const;

    /**
     * TODO
     */
    inline bool isOvercomplete() const;

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
     * Sets the sparsity weight for each entry of H, given as a matrix.
     */
    inline void setSparsity(const blissart::linalg::Matrix& s);

    /**
     * Sets the sparsity weight for each entry of H, given as a template.
     */
    void setSparsity(const SparsityTemplate& t);

    /**
     * Returns the sparsity weight for each entry of H as a matrix.
     */
    inline const blissart::linalg::Matrix& getSparsity() const;

    /**
     * Sets the sparsity weight for each entry of W(p)
     * given as a matrix.
     */
    inline void setWSparsity(unsigned int p, const blissart::linalg::Matrix& s);

    /**
     * Sets the sparsity weight for each entry of all W(p),
     * given as a template.
     */
    void setWSparsity(const SparsityTemplate& t);

    /**
     * Sets the continuity weight for each entry of H, given as a matrix.
     */
    inline void setContinuity(const blissart::linalg::Matrix& c);

    /**
     * Returns the continuity weight for each entry of H as a matrix.
     */
    inline const blissart::linalg::Matrix& getContinuity() const;

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
     * @param cf                the NMDCostFunction to minimize
     * @param maxSteps          maximum number of iteration steps
     * @param eps               if eps > 0.0, convergence of the approximation
     *                          is checked and once it is reached, iteration 
     *                          is stopped; setting eps > 0.0 may significantly
     *                          slow down computation and should not be used
     *                          in production code
     * @param observer          a ProgressObserver that is notified every 25
     *                          iteration steps
     */
    void decompose(NMDCostFunction cf, unsigned int maxSteps, double eps,
                   SparsityConstraint sparsity = NoSparsity, 
                   bool continuous = false,
                   ProgressObserver *observer = 0);


    /**
     * Returns the value of the cost function after the last iteration.
     */
    double getCfValue(NMDCostFunction cf, double beta = 0.0) const;


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


    /**
     * Performs NMD minimizing the beta divergence.
     * @param  maxSteps   maximum number of iteration steps
     * @param  eps        epsilon for the convergence criterion;
     *                    if (eps == 0), perform maxSteps iterations
     * @param  beta       the beta parameter for the divergence
     * @param  sparse     whether to enforce H sparsity
     * @param  continuous whether to enforce H continuity
     * @param  observer   a ProgressObserver that should be notified
     * Setting eps > 0 significantly slows down the NMD and should therefore 
     * not be used in production code.
     */
    void factorizeNMDBeta(unsigned int maxSteps, double eps, double beta,
                          SparsityConstraint sparsity = NoSparsity, 
                          bool continuous = false,
                          ProgressObserver *observer = 0);

    // A more efficient implementation of NMD-ED for 1 spectrum (NMF case)
    // for incomplete factorization.
    void factorizeNMFEDIncomplete(unsigned int maxSteps, double eps,
                                  ProgressObserver *observer = 0);

    // Sparse NMF minimizing Euclidean distance, measured using normalized
    // basis vectors (Eggert and Körner 2004).
    void factorizeNMFEDSparseNorm(unsigned int maxSteps, double eps,
                                  ProgressObserver *observer = 0);

protected:
    /**
     * Normalizes the matrices according to the given method.
     */
    void normalizeMatrices(MatrixNormalization method);


    // GPU version of computeApprox used only locally during factorization 
    // - might be unified with computeApprox() in the future
#ifdef HAVE_CUDA
    void computeApprox(blissart::linalg::GPUMatrix **w, 
        const blissart::linalg::GPUMatrix &h, 
        blissart::linalg::GPUMatrix *target);
#endif
    
    // The W update (common for factorizeNMFED and factorizeEDSparse).
    void factorizeNMFEDWUpdate(blissart::linalg::Matrix& w);

    // Calculate the update matrices (numerator/denominator) for the H update
    // (common for factorizeNMFED and factorizeNMFEDSparse).
    // Note that the actual update differs between sparse and non-sparse
    // versions.
    void calculateNMFEDHUpdate(blissart::linalg::Matrix& num,
                               blissart::linalg::Matrix& denom);

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
    
    // GPU version of the above. This might be unified with the above in the future.
#ifdef HAVE_CUDA
    void multWithShifted(const blissart::linalg::GPUMatrix &left, 
        const blissart::linalg::GPUMatrix &right,
        blissart::linalg::GPUMatrix *target,
        unsigned int shift) const;
#endif

    // Helper function that sets the negative elements of a matrix
    // to a small positive value.
    void ensureNonnegativity(blissart::linalg::Matrix& m, 
        double epsilon = 1e-9);

    // Normalizes Frobenius norm of H to unity.
    void normalizeHFrob();

    // Normalizes Euclidean norm of each W column to unity.
    void normalizeWColumnsEucl();

    inline bool useOvercompleteAlg() const;

    NMDAlgorithm                    _alg;
    MatrixNormalization             _norm;
    const blissart::linalg::Matrix& _v;
    blissart::linalg::Matrix _approx;
    blissart::linalg::Matrix*       _oldApprox;
    blissart::linalg::Matrix**      _w;
    bool                        _wConstant;
    bool*                       _wColConstant;
    unsigned int                _t;
    blissart::linalg::Matrix    _h;
    blissart::linalg::Matrix    _s;
    blissart::linalg::Matrix**  _wS;
    blissart::linalg::Matrix    _c;   
    unsigned int                _numSteps;
    double                      _absoluteError;
    double                      _relativeError;
    const double                _vFrob;
    unsigned int                _notificationDelay;


private:
    // Allocate helper variables if needed
    void factorizeNMFEDInitialize();
    // Deallocate these helper variables if previously allocated
    void factorizeNMFEDUninitialize();
    // Some helper variables to make factorizeNMFED() more efficient.
    // They are not allocated if another factorization method is used.
    blissart::linalg::Matrix *_hhT, *_wTw, 
                             *_wUpdateMatrixNum, *_wUpdateMatrixDenom;

    // Forbid copy constructor and operator=.
    Deconvolver(const Deconvolver&);
    Deconvolver& operator=(const Deconvolver&);
};


// Inlines


unsigned int Deconvolver::nrOfComponents() const
{
    return _h.rows();
}


inline void Deconvolver::setAlgorithm(Deconvolver::NMDAlgorithm a)
{
    _alg = a;
}


inline Deconvolver::NMDAlgorithm Deconvolver::getAlgorithm() const
{
    return _alg;
}


inline void 
Deconvolver::setNormalization(Deconvolver::MatrixNormalization norm)
{
    _norm = norm;
}


Deconvolver::MatrixNormalization Deconvolver::getNormalization() const
{
    return _norm;
}


bool Deconvolver::isOvercomplete() const
{
    return 
        // (m+n)r > mn?
        (_v.rows() + _v.cols()) * _h.rows() 
        > _v.rows() * _v.cols();
}


bool Deconvolver::useOvercompleteAlg() const
{
    return (_alg == Auto && isOvercomplete()) || _alg == Overcomplete;
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


const blissart::linalg::Matrix& Deconvolver::getH() const
{
    return _h;
}


void Deconvolver::setSparsity(const blissart::linalg::Matrix& s)
{
    assert(s.rows() == _h.rows() && s.cols() == _h.cols());
    _s = s;
}


const blissart::linalg::Matrix& Deconvolver::getSparsity() const
{
    return _s;
}


void Deconvolver::setWSparsity(unsigned int p,
    const blissart::linalg::Matrix& s)
{
    assert(s.rows() == _w[p]->rows() && s.cols() == _w[p]->cols());
    *_w[p] = s;
}


void Deconvolver::setContinuity(const blissart::linalg::Matrix& c)
{
    assert(c.rows() == _h.rows() && c.cols() == _h.cols());
    _c = c;
}


const blissart::linalg::Matrix& Deconvolver::getContinuity() const
{
    return _c;
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
