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


#include <blissart/nmf/Deconvolver.h>
#include <blissart/nmf/randomGenerator.h>
#include <blissart/linalg/generators/generators.h>
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


Deconvolver::Deconvolver(const Matrix &v, unsigned int r, unsigned int t,
                         Matrix::GeneratorFunction generator) :
    _v(v),
    _lambda(v.rows(), v.cols(), generators::zero),
    _w(new Matrix*[t]),
    _wConstant(false),
    _wColConstant(new bool[r]),
    _t(t),
    _h(r, v.cols(), generator),
    _numSteps(0),
    _absoluteError(-1),
    _relativeError(-1),
    _vFrob(_v.frobeniusNorm())
{
    assert(t <= v.cols());
    for (unsigned int l = 0; l < _t; ++l) {
        _w[l] = new Matrix(v.rows(), r);
    }
    for (unsigned int c = 0; c < r; ++c) {
        _wColConstant[c] = false;
    }
    generateW(generator);
}


Deconvolver::~Deconvolver()
{
    for (unsigned int l = 0; l < _t; ++l) {
        delete _w[l];
    }
    delete[] _w;
    delete[] _wColConstant;
}


void Deconvolver::generateW(Matrix::GeneratorFunction generator)
{
    for (unsigned int l = 0; l < _t; l++) {
        for (unsigned int i = 0; i < _w[l]->rows(); i++) {
            for (unsigned int j = 0; j < _w[l]->cols(); j++) {
                _w[l]->at(i,j) = generator(i, j);
            }
        }
    }
}


void Deconvolver::generateH(Matrix::GeneratorFunction generator)
{
    for (unsigned int i = 0; i < _h.rows(); i++) {
        for (unsigned int j = 0; j < _h.cols(); j++) {
            _h(i,j) = generator(i, j);
        }
    }
}


void Deconvolver::setW(unsigned int i, const Matrix& w)
{
    assert(w.cols() == _w[i]->cols() && w.rows() == _w[i]->rows());
    delete _w[i];
    _w[i] = new Matrix(w);
}


void Deconvolver::setH(const Matrix& h)
{
    assert(h.cols() == _h.cols() && h.rows() == _h.rows());
    _h = h;
}


void Deconvolver::factorizeKL(unsigned int maxSteps, double eps,
                              ProgressObserver *observer)
{
    Matrix vOverLambda(_v.rows(), _v.cols());
    Matrix hShifted(_h.rows(), _h.cols());
    Matrix vOverLambdaShifted(_v.rows(), _v.cols());
    Matrix hUpdate(_h.rows(), _h.cols());
    double *wtColSums = new double[_h.rows()];

#ifdef GNUPLOT
    std::ofstream os(GNUPLOT, std::ios_base::out | std::ios_base::trunc);
#endif

    _numSteps = 0;
    while (_numSteps < maxSteps) {

        // Update Lambda and V / Lambda
        computeLambda();
        _v.elementWiseDivision(_lambda, &vOverLambda);

        // Calculate update matrix for H by averaging the updates corresponding
        // to each W_t
        hUpdate.zero();
        vOverLambdaShifted = vOverLambda;
        for (unsigned int t = 0; t < _t; ++t) {
            // Precalculation of column-sums of W_t
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                wtColSums[i] = _w[t]->colSum(i);
            }

            // Calculate sum of updates
            for (unsigned int j = 0; j < _h.cols(); ++j) {
                for (unsigned int i = 0; i < _h.rows(); ++i) {
                    hUpdate(i, j) += 
                        Matrix::dotColCol(*_w[t], i, vOverLambdaShifted, j) / 
                        wtColSums[i];
                }
            }
            
            vOverLambdaShifted.shiftColumnsLeft();
        }

        // Apply average update to H
        for (unsigned int j = 0; j < _h.cols(); ++j) {
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                _h(i, j) *= hUpdate(i, j) / (double) _t;
            }
        }

        if (!_wConstant) {
            // Update Lambda and V / Lambda
            computeLambda();
            _v.elementWiseDivision(_lambda, &vOverLambda);

            // Update all W_t
            hShifted = _h;
            for (unsigned int t = 0; t < _t; ++t) {
                for (unsigned int j = 0; j < _w[t]->cols(); ++j) {
                    if (!_wColConstant[j]) {
                        // Precalculation of sum of row j of H
                        double hRowSum = hShifted.rowSum(j);
                        for (unsigned int i = 0; i < _w[t]->rows(); ++i) {
                            _w[t]->at(i, j) *= 
                                Matrix::dotRowRow(hShifted, j, vOverLambda, i) 
                                / hRowSum;
                        }
                    }
                }
                hShifted.shiftColumnsRight();
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

    delete[] wtColSums;
}


void Deconvolver::factorizeED(unsigned int maxSteps, double eps,
                              ProgressObserver *observer)
{
    Matrix hSum(_h.rows(), _h.cols());
    Matrix wUpdateMatrixNom(_v.rows(), _h.rows());
    Matrix wUpdateMatrixDenom(_v.rows(), _h.rows());
    Matrix hUpdateMatrixNom(_h.rows(), _h.cols());
    Matrix hUpdateMatrixDenom(_h.rows(), _h.cols());
    double denom;

#ifdef GNUPLOT
    std::ofstream os(GNUPLOT, std::ios_base::out | std::ios_base::trunc);
#endif

    _numSteps = 0;
    while (_numSteps < maxSteps) {

        // Compute approximation at the beginning and after the H update
        computeLambda();

        if (!_wConstant) {
            // Update all W[p]
            for (unsigned int p = 0; p < _t; ++p) {
                // Calculate V * (H shifted t spots to the right)^T 
                // (nominator of the update matrix)
                // In this case, zeros would be introduced in the first t rows
                // of the second factor. We can simulate this by considering
                // only the V columns starting from p.                
                _v.multWithMatrix(_h, &wUpdateMatrixNom,
                    // transpose H
                    false, true, 
                    // target dimension: MxR
                    _v.rows(), _v.cols() - p, _h.rows(),
                    0, p, 0, 0, 0, 0);
                // Calculate Lambda * (H shifted t spots to the right)^T 
                // (denominator of the update matrix)
                // The same as above.
                _lambda.multWithMatrix(_h, &wUpdateMatrixDenom,
                    false, true,
                    _v.rows(), _v.cols() - p, _h.rows(), 
                    0, p, 0, 0, 0, 0);
                // Efficient (difference-based) calculation of updated Lambda
                // (step 1: subtraction of old W[p]*H)
                // Due to Wang (2009)
                Matrix wpH(_v.rows(), _v.cols());
                // Fill W[p]*H with zeros in the first p columns
                for (unsigned int j = 0; j < p; ++j) {
                    for (unsigned int i = 0; i < wpH.rows(); ++i) {
                        wpH(i, j) = 0.0;
                    }
                }
                // Simulate multiplication with H shifted t spots to the right:
                // only use N - p columns of H for the matrix product
                // and store the result beginning at column p of W[p]*H
                // (for this reason W[p]*H had to be filled with zeros)
                _w[p]->multWithMatrix(_h, &wpH,
                    false, false,
                    _w[p]->rows(), _w[p]->cols(), _h.cols() - p,
                    0, 0, 0, 0, 0, p);
                // It is safe to overwrite Lambda, as it is not directly used 
                // in the update loop.
                _lambda.sub(wpH);
                // Finally, the update loop is simple now.
                for (unsigned int j = 0; j < _w[p]->cols(); ++j) {
                    if (!_wColConstant[j]) {
                        for (unsigned int i = 0; i < _w[p]->rows(); ++i) {
                            denom = wUpdateMatrixDenom(i, j);
                            if (denom <= 0.0) denom = 1e-9;
                            _w[p]->at(i, j) *= wUpdateMatrixNom(i, j) / denom;
                        }
                    }
                }
                // Calculate updated lambda, step 2
                // (addition of new W[p]*H)
                // Same procedure as above ...
                for (unsigned int j = 0; j < p; ++j) {
                    for (unsigned int i = 0; i < wpH.rows(); ++i) {
                        wpH(i, j) = 0.0;
                    }
                }
                _w[p]->multWithMatrix(_h, &wpH,
                    false, false,
                    _w[p]->rows(), _w[p]->cols(), _h.cols() - p,
                    0, 0, 0, 0, 0, p);
                _lambda.add(wpH);
                ensureNonnegativity(_lambda);
            }
        }

        // Calculate update matrix for H by averaging the updates corresponding
        // to each W[p]
        hSum.zero();
        for (unsigned int p = 0; p < _t; ++p) {
            // Simulate multiplication of W[p] with V (shifted p spots to the
            // left) by considering only the columns of V starting from p;
            // We do not fill with zeros here, because we ignore the rightmost
            // p columns of the nominator and denominator matrices in
            // the update loop below.
            _w[p]->multWithMatrix(_v, &hUpdateMatrixNom,
                // transpose W[p]
                true, false, 
                // target dimension: R x (N-p)
                _w[p]->cols(), _w[p]->rows(), _v.cols() - p,
                0, 0, 0, p, 0, 0);
            _w[p]->multWithMatrix(_lambda, &hUpdateMatrixDenom,
                true, false, 
                _w[p]->cols(), _w[p]->rows(), _v.cols() - p,
                0, 0, 0, p, 0, 0);

            for (unsigned int j = 0; j < _h.cols(); ++j) {
                for (unsigned int i = 0; i < _h.rows(); ++i) {
                    denom = hUpdateMatrixDenom(i, j);
                    // Avoid division by zero
                    if (denom <= 0.0) denom = 1e-9;
                    hSum(i, j) += _h(i, j) * hUpdateMatrixNom(i, j) / denom;
                }
            }
        }

        // Apply average update to H
        for (unsigned int j = 0; j < _h.cols(); ++j) {
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                _h(i, j) = hSum(i, j) / (double) _t;
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


void Deconvolver::computeLambda()
{
    Matrix hShifted(_h);
    Matrix whProd(_v.rows(), _v.cols());
    _lambda.zero();
    for (unsigned int l = 0; l < _t; ++l) {
        _w[l]->multWithMatrix(hShifted, &whProd);
        _lambda.add(whProd);
        hShifted.shiftColumnsRight();
    }
}


void Deconvolver::computeError()
{
    Matrix errorMatrix(_lambda);
    errorMatrix.sub(_v);
    _absoluteError = errorMatrix.frobeniusNorm();
    _relativeError = _absoluteError / _vFrob;
}


void Deconvolver::ensureNonnegativity(Matrix &m, double epsilon)
{
    for (unsigned int j = 0; j < m.cols(); ++j) {
        for (unsigned int i = 0; i < m.rows(); ++i) {
            if (m(i, j) <= 0.0) {
                m(i, j) = epsilon;
            }
        }
    }
}


} // namespace nmf

} // namespace blissart
