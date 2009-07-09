//
// $Id: Factorizer.h 855 2009-06-09 16:15:50Z alex $
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

#ifndef __BLISSART_NMF_FACTORIZER_H__
#define __BLISSART_NMF_FACTORIZER_H__


#include <common.h>
#include <blissart/linalg/Matrix.h>
#include <cassert>


namespace blissart {


// Forward declaration
class ProgressObserver;


namespace nmf {


/**
 * Performs non-negative matrix factorization using algorithms by Lee and Seung (2001).
 */
class LibNMF_API Factorizer
{
public:
    /**
     * Constructs a Factorizer object for the matrix V and the given dimensionality of the factorization.
     * Initializes the matrix factors with random matrices.
     * @param   v    a Matrix object to factorize
     * @param   r    the dimensionality of the factorization (i.e. number of columns of the first factor,
     *               number of rows of the second factor)
     */
    Factorizer(const blissart::linalg::Matrix& v, unsigned int r);

    /**
     * Specifies a value for the first factor.
     */
    inline void setFirst(const blissart::linalg::Matrix& w, bool constant = false);

    /**
     * Randomizes the first factor.
     */
    void randomizeFirst();

    /**
     * Returns the first factor.
     */
    inline const blissart::linalg::Matrix& getFirst() const;

    /**
     * Specifies a value for the second factor.
     */
    inline void setSecond(const blissart::linalg::Matrix& h);

    /**
     * Randomizes the second factor.
     */
    void randomizeSecond();

    /**
     * Returns the second factor.
     */
    inline const blissart::linalg::Matrix& getSecond() const;

    /**
     * Performs NMF factorization using euclidean distance measure.
     * Stops after the given number of iteration steps, or, if eps > 0,
     * if the relative error is smaller than eps.
     * If eps > 0, in each step the relative error is calculated, which
     * significantly slows down the NMF and should therefore not be used
     * in production code.
     */
    void factorizeDistance(unsigned int maxSteps, double eps,
                           ProgressObserver *observer = 0);

    /**
     * Performs NMF factorization using a generalized Kulback-Leibler divergence.
     * Stops after the given number of iteration steps, or, if eps > 0,
     * if the relative error is smaller than eps.
     * If eps > 0, in each step the relative error is calculated, which
     * significantly slows down the NMF and should therefore not be used
     * in production code.
     */
    void factorizeDivergence(unsigned int maxSteps, double eps,
                             ProgressObserver *observer = 0);

    /**
     * Performs NMF factorization by using a simple gradient descent.
     * Stops after the given number of iteration steps, or, if eps > 0,
     * if the relative error is smaller than eps.
     * If eps > 0, in each step the relative error is calculated, which
     * significantly slows down the NMF and should therefore not be used
     * in production code.
     */
    void factorizeGradient(unsigned int maxSteps, double eps,
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


private:
    // Forbid copy constructor and operator=.
    Factorizer(const Factorizer&);
    Factorizer& operator=(const Factorizer&);

    void computeError();

    const blissart::linalg::Matrix& _v;
    blissart::linalg::Matrix        _w;
    bool                        _wConstant;
    blissart::linalg::Matrix        _h;
    unsigned int                _numSteps;
    double                      _absoluteError;
    double                      _relativeError;
    const double                _vFrob;
};


// Inlines

void Factorizer::setFirst(const blissart::linalg::Matrix& w, bool constant)
{
    assert(w.cols() == _w.cols() && w.rows() == _w.rows());
    _w = w;
    _wConstant = constant;
}


const blissart::linalg::Matrix& Factorizer::getFirst() const
{
    return _w;
}


void Factorizer::setSecond(const blissart::linalg::Matrix& h)
{
    assert(h.cols() == _h.cols() && h.rows() == _h.rows());
    _h = h;
}


const blissart::linalg::Matrix& Factorizer::getSecond() const
{
    return _h;
}


double Factorizer::absoluteError()
{
    if (_absoluteError < 0)
        computeError();
    return _absoluteError;
}


double Factorizer::relativeError()
{
    if (_relativeError < 0)
        computeError();
    return _relativeError;
}


unsigned int Factorizer::numSteps() const
{
    return _numSteps;
}


} // namespace nmf

} // namespace blissart


#endif // __BLISSART_NMF_FACTORIZER_H__
