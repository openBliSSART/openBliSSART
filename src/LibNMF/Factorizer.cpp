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


#include <blissart/nmf/Factorizer.h>
#include <blissart/nmf/randomGenerator.h>
#include <blissart/ProgressObserver.h>

// Uncomment the following line if you want to generate output suitable for
// gnuplot during factorization.
//#define GNUPLOT "relative.plt"

#ifdef GNUPLOT
# include <iostream>
# include <fstream>
#endif


using namespace blissart::linalg;


namespace blissart {

namespace nmf {


Factorizer::Factorizer(const Matrix &v, unsigned int r) :
    _v(v),
    _w(v.rows(), r, uniformRandomGenerator),
    _wConstant(false),
    _h(r, v.cols(), uniformRandomGenerator),
    _numSteps(0),
    _absoluteError(-1),
    _relativeError(-1),
    _vFrob(_v.frobeniusNorm())
{
}


void Factorizer::randomizeFirst()
{
    for (unsigned int i = 0; i < _w.rows(); i++) {
        for (unsigned int j = 0; j < _w.cols(); j++) {
            _w(i,j) = uniformRandomGenerator(i, j);
        }
    }
}


void Factorizer::randomizeSecond()
{
    for (unsigned int i = 0; i < _h.rows(); i++) {
        for (unsigned int j = 0; j < _h.cols(); j++) {
            _h(i,j) = uniformRandomGenerator(i, j);
        }
    }
}


void Factorizer::factorizeDistance(unsigned int maxSteps, double eps,
                                   ProgressObserver *observer)
{
    Matrix wTransposed(_w.cols(), _w.rows());

    Matrix wTw(_w.cols(), _w.cols());
    Matrix hUpdateMatrixNom(_h.rows(), _h.cols());
    Matrix hUpdateMatrixDenom(_h.rows(), _h.cols());

    Matrix hhT(_h.rows(), _h.rows());
    Matrix wUpdateMatrixNom(_w.rows(), _w.cols());
    Matrix wUpdateMatrixDenom(_w.rows(), _w.cols());

#ifdef GNUPLOT
    std::ofstream os(GNUPLOT, std::ios_base::out | std::ios_base::trunc);
#endif

    _numSteps = 0;
    while (_numSteps < maxSteps) {
        // Calculate the update matrix for H
        _w.transpose(&wTransposed);
        wTransposed.multWithMatrix(_v, &hUpdateMatrixNom);
        wTransposed.multWithMatrix(_w, &wTw);
        wTw.multWithMatrix(_h, &hUpdateMatrixDenom);

        // Perform element-wise update of H
        for (unsigned int i = 0; i < _h.rows(); i++) {
            for (unsigned j = 0; j < _h.cols(); j++) {
                _h(i,j) *= hUpdateMatrixNom(i,j) / hUpdateMatrixDenom(i,j);
            }
        }

        // Calculate the update matrix for W
        //_h.transpose(&hTransposed);
        _v.multWithTransposedMatrix(_h, &wUpdateMatrixNom);
        _h.multWithTransposedMatrix(_h, &hhT);
        _w.multWithMatrix(hhT, &wUpdateMatrixDenom);

        // Perform element-wise update of W
        for (unsigned int i = 0; i < _w.rows(); i++) {
            for (unsigned j = 0; j < _w.cols(); j++) {
                _w(i,j) *= wUpdateMatrixNom(i,j) / wUpdateMatrixDenom(i,j);
            }
        }

#ifdef GNUPLOT
        computeError();
        os << _relativeError << std::endl;
#endif

        if (eps > 0.0) {
            computeError();
            if (_relativeError < eps) break;
        }

        ++_numSteps;

        // Call the ProgressObserver every once in a while (if applicable).
        if (observer && _numSteps % 25 == 0)
            observer->progressChanged((float)_numSteps / (float)maxSteps);
    }
    // Final call to the ProgressObserver (if applicable).
    if (observer)
        observer->progressChanged(1.0f);
}


void Factorizer::factorizeDivergence(unsigned int maxSteps, double eps,
                                     ProgressObserver *observer)
{
    Matrix wh(_w.rows(), _h.cols());
    Matrix v_div_wh(_w.rows(), _h.cols());

    double* wColSums = new double[_h.rows()];
    double hRowSum;

#ifdef GNUPLOT
    std::ofstream os(GNUPLOT, std::ios_base::out | std::ios_base::trunc);
#endif

    _numSteps = 0;
    while (_numSteps < maxSteps) {
        // Compute matrix WH
        _w.multWithMatrix(_h, &wh);

        // Precalculation of v_div_wh
        _v.elementWiseDivision(wh, &v_div_wh);

        // Precalculation of column-sums of W
        for (unsigned int i = 0; i < _h.rows(); i++) {
            wColSums[i] = _w.colSum(i);
        }

        // Perform elementwise update of H
        for (unsigned int j = 0; j < _h.cols(); j++) {
            for (unsigned int i = 0; i < _h.rows(); i++) {
                _h(i, j) *= Matrix::dotColCol(_w, i, v_div_wh, j) / wColSums[i];
            }
        }

        if (!_wConstant) {
            // Re-compute matrix WH
            _w.multWithMatrix(_h, &wh);

            // Precalculation of v_div_wh (on updated matrices)
            _v.elementWiseDivision(wh, &v_div_wh);

            // Perform elementwise update of W
            for (unsigned int j = 0; j < _w.cols(); j++) {
                // Precalculation of sum of row j of H
                hRowSum = _h.rowSum(j);
                for (unsigned int i = 0; i < _w.rows(); i++) {
                    _w(i,j) *= Matrix::dotRowRow(_h, j, v_div_wh, i) / hRowSum;
                }
            }
        }

#ifdef GNUPLOT
        computeError();
        os << _relativeError << std::endl;
#endif

        if (eps > 0.0) {
            computeError();
            if (_relativeError < eps) break;
        }

        ++_numSteps;

        // Call the ProgressObserver every once in a while (if applicable).
        if (observer && _numSteps % 25 == 0)
            observer->progressChanged((float)_numSteps / (float)maxSteps);
    }
    // Final call to the ProgressObserver (if applicable).
    if (observer)
        observer->progressChanged(1.0f);

    delete[] wColSums;
}


void Factorizer::factorizeGradient(unsigned int maxSteps, double eps,
                                   ProgressObserver *observer)
{
    Matrix whv(_w.rows(), _h.cols());
    Matrix delta_w(_w.rows(), _w.cols());
    Matrix delta_h(_h.rows(), _h.cols());

#ifdef GNUPLOT
    std::ofstream os(GNUPLOT, std::ios_base::out | std::ios_base::trunc);
#endif

    _numSteps = 0;
    while (_numSteps < maxSteps) {
        // Compute matrix WH-V
        _w.multWithMatrix(_h, &whv);
        whv.sub(_v);

        // Gradients
        delta_w = whv * _h.transposed();
        delta_h = _w.transposed() * whv;

        // Gradient descent
        _w.sub(1e-4 * delta_w);
        _h.sub(1e-4 * delta_h);

        // Enforce non-negative constraints.
        _w.eliminateNegativeElements();
        _h.eliminateNegativeElements();

#ifdef GNUPLOT
        computeError();
        os << _relativeError << std::endl;
#endif

        if (eps > 0.0) {
            computeError();
            if (_relativeError < eps) break;
        }

        ++_numSteps;

        // Call the ProgressObserver every once in a while (if applicable).
        if (observer && _numSteps % 25 == 0)
            observer->progressChanged((float)_numSteps / (float)maxSteps);
    }
    // Final call to the ProgressObserver (if applicable).
    if (observer)
        observer->progressChanged(1.0f);
}


void Factorizer::computeError()
{
    // TODO: Multiplying W by H and possibly also subtracting V has probably
    // already been done by the caller in some way.

    // Calculate errorMatrix = W * H - V
    Matrix errorMatrix(_w.rows(), _h.cols());
    _w.multWithMatrix(_h, &errorMatrix);
    errorMatrix.sub(_v);
    _absoluteError = errorMatrix.frobeniusNorm();
    _relativeError = _absoluteError / _vFrob;
}


} // namespace nmf

} // namespace blissart
