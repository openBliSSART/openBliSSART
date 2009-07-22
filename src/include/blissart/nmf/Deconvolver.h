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
#include <cassert>


namespace blissart {


// Forward declaration
class ProgressObserver;


namespace nmf {


/**
 * Performs non-negative matrix deconvolution using Smaragdis' algorithm (2004).
 */
class LibNMF_API Deconvolver
{
public:
    /**
     * Constructs a Deconvolver object for the matrix V.
     * Initializes the matrix factors with random matrices.
     * @param   v    a Matrix object
     * @param   r    the dimensionality of the factorization
     * @param   t    the desired number of W matrices
     */
    Deconvolver(const blissart::linalg::Matrix& v, unsigned int r, unsigned int t);

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
     * Randomizes all W matrices.
     */
    void randomizeW();

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
    void randomizeH();

    /**
     * Returns the current value of Lambda (approximation of V).
     */
    inline const blissart::linalg::Matrix& getLambda() const;
    
    /**
     * Updates the value of Lambda (approximation of V).
     */
    void computeLambda();

    /**
     * Performs NMD using a generalized Kulback-Leibler divergence.
     * Stops after the given number of iteration steps, or, if eps > 0,
     * if the relative error is smaller than eps.
     * If eps > 0, in each step the relative error is calculated, which
     * significantly slows down the NMF and should therefore not be used
     * in production code.
     */
    void factorize(unsigned int maxSteps, double eps,
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


protected:
    // Computes error of current approximation.
    void computeError();

    const blissart::linalg::Matrix& _v;
    blissart::linalg::Matrix        _lambda;
    blissart::linalg::Matrix**      _w;
    bool                        _wConstant;
    bool*                       _wColConstant;
    unsigned int                _t;
    blissart::linalg::Matrix        _h;
    unsigned int                _numSteps;
    double                      _absoluteError;
    double                      _relativeError;
    const double                _vFrob;


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


const blissart::linalg::Matrix& Deconvolver::getH() const
{
    return _h;
}


const blissart::linalg::Matrix& Deconvolver::getLambda() const
{
    return _lambda;
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


} // namespace nmf

} // namespace blissart


#endif // __BLISSART_NMF_DECONVOLVER_H__
