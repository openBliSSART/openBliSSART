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


#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/generators/generators.h>
#include <blissart/BinaryReader.h>
#include <blissart/BinaryWriter.h>

#include <cstring>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <cassert>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_CBLAS_H
extern "C" {
#include <cblas.h>
}
#endif


namespace blissart {

namespace linalg {


Matrix::Matrix(const Matrix& other) :
    _rows(other._rows),
    _cols(other._cols),
    _data(new double[other._rows * other._cols])
{
    debug_assert(_rows > 0 && _cols > 0);

    memcpy(_data, other._data, _rows * _cols * sizeof(double));
}


Matrix::Matrix(unsigned int rows, unsigned int cols) :
    _rows(rows),
    _cols(cols),
    _data(new double[rows * cols])
{
    debug_assert(_rows > 0 && _cols > 0);
}


Matrix::Matrix(unsigned int rows, unsigned int cols, const double* data,
               bool useRawPointer) :
    _rows(rows),
    _cols(cols)
{
    debug_assert(_rows > 0 && _cols > 0);

    if (useRawPointer) {
        _data = const_cast<double *>(data);
    } else {
        _data = new double[_rows * _cols];
#ifndef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++)
        for (unsigned int j = 0; j < _cols; j++, data++)
            (*this)(i, j) = *data;
#else
    memcpy(_data, data, _rows * _cols * sizeof(double));
#endif
    }
}


Matrix::Matrix(unsigned int rows, unsigned int cols,
               double (*generator)(unsigned int i, unsigned int j)) :
    _rows(rows),
    _cols(cols),
    _data(new double[rows * cols])
{
    debug_assert(_rows > 0 && _cols > 0);

    for (unsigned int i = 0; i < _rows; i++)
        for (unsigned int j = 0; j < _cols; j++)
            setAt(i, j, generator(i, j));
}


Matrix::Matrix(const std::string& fileName) :
    _data(0)
{
    do {
        std::ifstream fis(fileName.c_str(), std::ios::in | std::ios::binary);
        if (fis.fail())
            break;

        BinaryReader br(fis, BinaryReader::LittleEndian);

        uint32_t flag;
        br >> flag;
        if (flag != 2)
            break;

        br >> _rows;
        br >> _cols;
        if (_rows <= 0 || _cols <= 0 || _rows * _cols <= 0)
            break;

        _data = new double[_rows * _cols];
#ifdef ISEP_ROW_MAJOR
        bool ok = true;
        double* dataPtr = _data;
        for (unsigned int j = 0; ok && j < _cols; ++j) {
            for (unsigned int i = 0; i < _rows; ++i) {
                if (br.fail() || br.eof()) {
                    ok = false;
                    break;
                }
                br >> at(i, j);
            }
        }
        if (!ok) break;
#else
        if (br.readDoubles(_data, _rows * _cols) != _rows * _cols)
            break;
#endif

        // Everything's ok, so return at this point.
        return;
    } while (false);

    // Something went wrong.
    if (_data)
        delete _data;
    std::string msg("Error while reading matrix data from file ");
    msg.append(fileName);
    throw std::runtime_error(msg);
}


Matrix::~Matrix()
{
    delete[] _data;
}


ColVector Matrix::nthColumn(unsigned int n) const
{
    debug_assert(n < _cols);

#ifndef ISEP_ROW_MAJOR
    return ColVector(_rows, _data + n * _rows);
#else
    ColVector result(_rows);
    for (unsigned int i = 0; i < _rows; i++)
        result(i) = this->at(i, n);
    return result;
#endif
}


RowVector Matrix::nthRow(unsigned int n) const
{
    debug_assert(n < _rows);

#ifndef ISEP_ROW_MAJOR
    RowVector result(_cols);
    for (unsigned int i = 0; i < _cols; i++)
        result(i) = this->at(n, i);
    return result;
#else
    return RowVector(_cols, _data + n * _cols);
#endif
}


void Matrix::nthRow2DoubleArray(unsigned int n, double* da) const
{
    debug_assert(n < _rows);

#ifndef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _cols; i++, da++)
        *da = this->at(n, i);
#else
    memcpy(da, _data + n * _cols, _cols * sizeof(double));
#endif
}


Matrix Matrix::upToAndIncludingRow(unsigned int row) const
{
    debug_assert(row <= _rows);

#ifndef ISEP_ROW_MAJOR
    Matrix result(row + 1, _cols);
    for (unsigned int i = 0; i < _cols; i++) {
        for (unsigned int j = 0; j <= row; j++)
            result(j, i) = this->at(j, i);
    }
    return result;
#else
    return Matrix(row + 1, _cols, _data);
#endif
}


void Matrix::setColumn(unsigned int col, const ColVector& cv)
{
    debug_assert(_rows == cv.dim() && col < _cols);

    for (unsigned int i = 0; i < _rows; i++)
        setAt(i, col, cv.at(i));
}


void Matrix::setRow(unsigned int row, const RowVector& rv)
{
    debug_assert(_cols == rv.dim() && row < _rows);

    for (unsigned int i = 0; i < _cols; i++)
        setAt(row, i, rv.at(i));
}


void Matrix::copyRow(unsigned int dstRow, unsigned int srcRow, const Matrix& other)
{
    debug_assert(dstRow < _rows &&
                 srcRow < other._rows &&
                 _cols == other._cols);

#ifndef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _cols; i++)
        setAt(dstRow, i, other.at(srcRow, i));
#else
    memcpy(_data + dstRow * _cols, other._data + srcRow * _cols,
           _cols * sizeof(double));
#endif
}


bool Matrix::isSymmetric() const
{
    if (_rows != _cols)
        return false;

    for (unsigned int i = 0; i < _rows; i++)
        for (unsigned int j = 0; j < _cols; j++)
            if (this->at(i, j) != this->at(j, i))
                return false;

    return true;
}


bool Matrix::isQuadratic() const
{
    return (_rows == _cols);
}


double Matrix::frobeniusNorm() const
{
    double result = 0.0;
    double* dataPtr = _data;
    double* dataEndPtr = _data + _rows * _cols;
    for (; dataPtr < dataEndPtr; ++dataPtr) {
        result += (*dataPtr) * (*dataPtr);
    }
    return sqrt(result);
}


void Matrix::multWithMatrix(const Matrix& other, Matrix* target) const
{
    debug_assert(_cols == other._rows &&
                 target->_rows == _rows &&
                 target->_cols == other._cols &&
                 target != this &&
                 target != &other);

#ifdef HAVE_CBLAS_H
#  ifdef ISEP_ROW_MAJOR
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,   // A (m x k)
                CblasNoTrans,   // B (k x n)
                target->_rows,  // m
                target->_cols,  // n
                this->_cols,    // k
                1.0,            // alpha
                this->_data,
                this->_cols,    // lda
                other._data,
                other._cols,    // ldb
                0.0,            // beta
                target->_data,
                target->_cols); // ldc
#  else // !ISEP_ROW_MAJOR
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,   // A (m x k)
                CblasNoTrans,   // B (k x n)
                target->_rows,  // m
                target->_cols,  // n
                this->_cols,    // k
                1.0,            // alpha
                this->_data,
                this->_rows,    // lda
                other._data,
                other._rows,    // ldb
                0.0,            // beta
                target->_data,
                target->_rows); // ldc
#  endif // ISEP_ROW_MAJOR
#else // !HAVE_CBLAS_H
#  if defined(ISEP_FAST_MATRIX_MULT) && !defined(ISEP_ROW_MAJOR)
    double* target_data_ptr = target->_data;
    for (unsigned int j = 0; j < target->_cols; j++) {
        for (unsigned int i = 0; i < target->_rows; i++) {
            double* this_data_ptr = _data + i;
            double* other_data_ptr = other._data + j * other._rows;
            *target_data_ptr = 0;
            for (unsigned int k = 0; k < _cols; k++) {
                *target_data_ptr += *this_data_ptr * *other_data_ptr;
                other_data_ptr++;
                this_data_ptr += _rows;
            }
            target_data_ptr++;
        }
    }
#  else // !ISEP_FAST_MATRIX_MULT || ISEP_ROW_MAJOR
#    ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < target->_rows; i++) {
        for (unsigned int j = 0; j < target->_cols; j++) {
#    else // !ISEP_ROW_MAJOR
    for (unsigned int j = 0; j < target->_cols; j++) {
        for (unsigned int i = 0; i < target->_rows; i++) {
#    endif // ISEP_ROW_MAJOR
            (*target)(i,j) = 0;
            for (unsigned int k = 0; k < _cols; k++) {
                (*target)(i,j) += this->at(i,k) * other.at(k,j);
            }
        }
    }
#  endif // ISEP_FAST_MATRIX_MULT && !ISEP_ROW_MAJOR
#endif // HAVE_CBLAS_H
}


Matrix Matrix::multWithTransposedMatrix(const Matrix& other) const
{
    debug_assert(_cols == other._cols);

    Matrix result(_rows, other._rows);
    multWithTransposedMatrix(other, &result);
    return result;
}


void Matrix::multWithTransposedMatrix(const Matrix& other, Matrix* target) const
{
    debug_assert(_cols == other._cols &&
                 target != this &&
                 target != &other);

#ifdef HAVE_CBLAS_H
#  ifdef ISEP_ROW_MAJOR
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,   // A (m x k)
                CblasTrans,     // B (k x n)
                target->_rows,  // m
                target->_cols,  // n
                this->_cols,    // k
                1.0,            // alpha
                this->_data,
                this->_cols,    // lda
                other._data,
                other._cols,    // ldb
                0.0,            // beta
                target->_data,
                target->_cols); // ldc
#  else // !ISEP_ROW_MAJOR
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,   // A (m x k)
                CblasTrans,     // B (k x n)
                target->_rows,  // m
                target->_cols,  // n
                this->_cols,    // k
                1.0,            // alpha
                this->_data,
                this->_rows,    // lda
                other._data,
                other._rows,    // ldb
                0.0,            // beta
                target->_data,
                target->_rows); // ldc
#  endif // ISEP_ROW_MAJOR
#else // !HAVE_CBLAS_H
#  if defined(ISEP_FAST_MATRIX_MULT) && !defined(ISEP_ROW_MAJOR)
    double* target_data_ptr = target->_data;
    for (unsigned int j = 0; j < target->_cols; j++) {
        for (unsigned int i = 0; i < target->_rows; i++) {
            double* this_data_ptr = _data + i;
            double* other_data_ptr = other._data + j;
            *target_data_ptr = 0;
            for (unsigned int k = 0; k < _cols; k++) {
                *target_data_ptr += *this_data_ptr * *other_data_ptr;
                other_data_ptr += other._rows;
                this_data_ptr += _rows;
            }
            target_data_ptr++;
        }
    }
#  else // !ISEP_FAST_MATRIX_MULT || ISEP_ROW_MAJOR
#    ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < target->_rows; i++) {
        for (unsigned int j = 0; j < target->_cols; j++) {
#    else // !ISEP_ROW_MAJOR
    for (unsigned int j = 0; j < target->_cols; j++) {
        for (unsigned int i = 0; i < target->_rows; i++) {
#    endif // ISEP_ROW_MAJOR
            (*target)(i,j) = 0;
            for (unsigned int k = 0; k < _cols; k++) {
                (*target)(i,j) += this->at(i,k) * other.at(j,k);
            }
        }
    }
#  endif // ISEP_FAST_MATRIX_MULT && ISEP_ROW_MAJOR
#endif
}


void Matrix::elementWiseDivision(const Matrix& other, Matrix* target) const
{
    debug_assert(_rows == other._rows &&
                 _cols == other._cols &&
                 target->_rows == _rows &&
                 target->_cols == _cols);

    double *p1 = _data, *p1Max = _data + _rows * _cols;
    double *p2 = other._data;
    double *p3 = target->_data;
    while (p1 < p1Max)
        *(p3++) = *(p1++) / *(p2++);
}


void Matrix::transpose(Matrix* target) const
{
    debug_assert(target->_rows == _cols &&
                 target->_cols == _rows &&
                 target != this);

#ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
#else
    for (unsigned int j = 0; j < _cols; j++) {
        for (unsigned int i = 0; i < _rows; i++) {
#endif
            (*target)(j,i) = this->at(i,j);
        }
    }
}


Matrix Matrix::transposed() const
{
    Matrix result(_cols, _rows);
    transpose(&result);
    return result;
}


void Matrix::shiftColumnsLeft()
{
#ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols - 1; j++) {
            (*this)(i, j) = (*this)(i, j + 1);
        }
        (*this)(i, _cols - 1) = 0.0;
    }
#else // !ISEP_ROW_MAJOR
    double *pDst = _data;
    double *pEnd = _data + _rows * _cols;
    if (_cols > 1) {
        double *pSrc = _data + _rows;
        double *pLastCol = _data + _rows * (_cols - 1);
        while (pDst < pLastCol) {
            *pDst = *pSrc;
            pSrc++;
            pDst++;
        }
    }
    while (pDst < pEnd) {
        *pDst = 0.0;
        pDst++;
    }
#endif
}


void Matrix::shiftColumnsRight()
{
#ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++) {
        for (int j = (int) _cols - 1; j > 0; j--) {
            (*this)(i, j) = (*this)(i, j - 1);
        }
        (*this)(i, 0) = 0.0;
    }
#else // !ISEP_ROW_MAJOR
    double *pDst = _data + _rows * _cols - 1;
    if (_cols > 1) {
        double *pSrc = pDst - _rows;
        double *pSecondCol = _data + _rows;
        while (pDst >= pSecondCol) {
            *pDst = *pSrc;
            pSrc--;
            pDst--;
        }
    }
    while (pDst >= _data) {
        *pDst = 0.0;
        pDst--;
    }
#endif
}


void Matrix::add(const Matrix& other)
{
    debug_assert(_rows == other._rows &&
                 _cols == other._cols);

#ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
#else
    for (unsigned int j = 0; j < _cols; j++) {
        for (unsigned int i = 0; i < _rows; i++) {
#endif
            (*this)(i, j) += other.at(i, j);
        }
    }
}


void Matrix::sub(const Matrix& other)
{
    debug_assert(_rows == other._rows &&
                 _cols == other._cols);

#ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
#else
    for (unsigned int j = 0; j < _cols; j++) {
        for (unsigned int i = 0; i < _rows; i++) {
#endif
            (*this)(i, j) -= other.at(i, j);
        }
    }
}


void Matrix::zero()
{
    double *pEnd = _data + _rows * _cols;
    for (double* pData = _data; pData < pEnd; ++pData) {
        *pData = 0.0;
    }
}


unsigned int Matrix::gaussElimination(bool reducedRowEchelonForm)
{
    unsigned int swappedRows = 0;

    for (unsigned int i=0; i<_rows-1; i++) {
        // Spalten-Pivot-Suche
        unsigned int maxIndex = i;
        double max = fabs(this->at(i,i));
        for (unsigned int j=i+1; j<_rows; j++) {
            const double temp = fabs(this->at(j,i));
            if (temp > max) {
                maxIndex = j;
                max = temp;
            }
        }
        if (max == 0.0)
            throw std::runtime_error("Matrix under-determined!");

        // Zeilen tauschen
        if (maxIndex != i) {
            for (unsigned int j=i; j<_cols; j++) {
                double temp = this->at(i,j);
                (*this)(i,j) = this->at(maxIndex,j);
                (*this)(maxIndex,j) = temp;
            }
            swappedRows++;
        }

        // Die unteren Eintraege der aktuellen Spalte nullieren
        const double pivot = this->at(i,i);
        for (unsigned int j=i+1; j<_rows; j++) {
            const double f = this->at(j,i) / pivot;
            if (f != 0) {
                for (unsigned int k=i; k<_cols; k++) {
                    (*this)(j,k) -= this->at(i,k) * f;
                }
            }
        }
    }

    if (reducedRowEchelonForm) {
        for (int i=_rows-1; i>=0; i--) {
            // Die aktuelle Zeile skalieren
            const double f = 1.0 / this->at(i,i);
            for (unsigned int j=i; j<_cols; j++)
                this->at(i,j) *= f;
            // Die oberen Eintraege der aktuellen Spalte nullieren
            for (int j=i-1; j>=0; j--) {
                const double f = this->at(j,i) / this->at(i,i);
                for (unsigned int k=j; k<_cols; k++) {
                    this->at(j,k) -= this->at(i,k) * f;
                }
            }
        }
    }

    return swappedRows;
}


void Matrix::linearSolve(const Matrix& m, const ColVector& b, ColVector* target)
{
    debug_assert(m._rows == b.dim() && m._cols == target->dim());

    // Initialize LSE
    Matrix A(m._rows, m._cols+1);
    for (unsigned int i=0; i<m._rows; i++) {
        for (unsigned int j=0; j<m._cols; j++) {
            A(i,j) = m.at(i,j);
        }
        A(i,m._cols) = b.at(i);
    }

    // Solve LSE
    A.gaussElimination();
    for (unsigned int y=A._rows-1; y>=0; y--) {
        (*target)(y) = A(y,m._cols);
        for (unsigned int i=y+1; i<m._cols; i++) {
            (*target)(y) -= A(y,i) * target->at(i);
        }
        (*target)(y) /= A(y,y);
        // This check is necessary since we are using unsigned ints.
        // Otherwise the decrement of y would lead to overflow.
        if (y==0) break;
    }
}


double Matrix::determinant(const bool triangularHint)
{
    debug_assert(_rows == _cols);

    if (_rows == 1) {
        return at(0,0);
    } else if (_rows == 2) {
        return at(0,0) * at(1,1) - at(0,1) * at(1,0);
    } else if (_rows == 3) {
        return   at(0,0) * at(1,1) * at(2,2)
               + at(0,1) * at(1,2) * at(2,0)
               + at(0,2) * at(1,0) * at(2,1)
               - at(0,2) * at(1,1) * at(2,0)
               - at(0,1) * at(1,0) * at(2,2)
               - at(0,0) * at(1,2) * at(2,1);
    }

    // Compute via upper triangular matrix
    Matrix tmp(*this);
    const unsigned int swappedRows = triangularHint ? 0 : tmp.gaussElimination();
    double result = (swappedRows & 1) ? -tmp(0,0) : tmp(0,0);
    for (unsigned int i = 1; i < tmp._rows; i++)
        result *= tmp(i,i);
    return result;
}


double Matrix::trace() const
{
    debug_assert(isQuadratic());

    double result = 0;
    for (unsigned int i = 0; i < _rows; i++)
        result += this->at(i,i);

    return result;
}


Matrix::EigenPairs Matrix::eigenPairs(unsigned int maxNrOfEigenPairs,
                                      unsigned int maxIter, double prec) const
{
    assert(maxNrOfEigenPairs <= _rows);
    assert(prec > 0);

    EigenPairs eigenp;
    Matrix m_prime(*this);

    // Since the matrix class deals with datatype double it is assured that the
    // this matrix is real. However, it must be assured that it's also symmetric.
    assert(m_prime.isSymmetric());

    // XXX: For the power iteration the condition |lambda_1| > |lambda_2| > ...
    // must hold. The caller can determine if this was the case if the number
    // of computed eigenvalues equals the row- or column-dimension of the
    // given symmetric matrix.

    // Compute all eigenvalues if maxNrOfEigenPairs was given as 0.
    if (maxNrOfEigenPairs == 0)
        maxNrOfEigenPairs = m_prime.rows();

    eigenp.clear();
    for (unsigned int i = 0; i < maxNrOfEigenPairs; i++) {
        double lambda = -1, last_lambda = -1;
        ColVector v(m_prime.rows(), generators::random);
        v.normalize();
        // Power iteration:
        for (unsigned int mi = 0; mi < maxIter; mi++) {
            ColVector vn = m_prime * v;
            lambda = v.transposed() * vn;
            vn.normalize();
            if (mi > 0 && fabs(lambda - last_lambda) < prec) {
                v = vn;
                break;
            }
            last_lambda = lambda;
            v = vn;
        }
        eigenp.push_back(std::pair<double, ColVector>(lambda, v));
        // Deflation:
        if (i < maxNrOfEigenPairs-1)
            m_prime.sub(lambda * v * v.transposed());
        // Because of |lambda_1| > |lambda_2| > ... we can stop if the
        // eigenvalue equals zero.
        if (lambda == 0)
            break;
    }

    return eigenp;
}


void Matrix::eliminateNegativeElements()
{
    for (unsigned int i = 0; i < _rows * _cols; i++) {
        if (_data[i] < 0)
            _data[i] = 0;
    }
}


double Matrix::colSum(unsigned int column) const
{
    /// XXX: CBLAS is neither faster nor slower, so I'm leaving the code here
    ///      yet commented out.
//#ifdef HAVE_CBLAS_H
//    // Since BLAS doesn't provide a function for summing up a vector's elements,
//    // we have to perform a little trick at this point, i.e. computing the
//    // dot-product of the given column and a vector whose elements are all 1.
//    const double t = 1.0;
//#  ifdef ISEP_ROW_MAJOR
//    return cblas_ddot(_rows, _data + column, _cols, &t, 0);
//#  else // !ISEP_ROW_MAJOR
//    return cblas_ddot(_rows, _data + column * _rows, 1, &t, 0);
//#  endif // ISEP_ROW_MAJOR
//#else // !HAVE_CBLAS_H
    double result = 0.0;
#  ifdef ISEP_ROW_MAJOR
    double *pData = _data + column;
    double *pDataEnd = _data + _rows * _cols;
    while (pData < pDataEnd) {
        result += *pData;
        pData += _cols;
    }
#  else // !ISEP_ROW_MAJOR
    double *pColumn = _data + column * _rows;
    double *pColumnEnd = pColumn + _rows;
    while (pColumn < pColumnEnd) {
        result += *(pColumn++);
    }
#  endif // ISEP_ROW_MJAOR
    return result;
//#endif // HAVE_CBLAS_H
}


double Matrix::rowSum(unsigned int row) const
{
    /// XXX: CBLAS is neither faster nor slower, so I'm leaving the code here
    ///      yet commented out.
//#ifdef HAVE_CBLAS_H
//    // Please see colSum(...) for an explanation of this implementation.
//    const double t = 1.0;
//#  ifdef ISEP_ROW_MAJOR
//    return cblas_ddot(_cols, _data + row * _cols, 1, &t, 0);
//#  else // !ISEP_ROW_MAJOR
//    return cblas_ddot(_cols, _data + row, _rows, &t, 0);
//#  endif // ISEP_ROW_MAJOR
//#else // !HAVE_CBLAS_H
    double result = 0.0;
#  ifdef ISEP_ROW_MAJOR
    double *pRow = _data + row * _cols;
    double *pRowEnd = pRow + _cols;
    while (pRow < pRowEnd) {
        result += *(pRow++);
    }
#  else // !ISEP_ROW_MAJOR
    double *pData = _data + row;
    double *pDataEnd = _data + _rows * _cols;
    while (pData < pDataEnd) {
        result += *pData;
        pData += _rows;
    }
#  endif // ISEP_ROW_MAJOR
    return result;
//#endif // HAVE_CBLAS_H
}


double Matrix::dotColCol(const Matrix &a, unsigned int aCol,
                         const Matrix &b, unsigned int bCol)
{
    debug_assert(a._rows == b._rows &&
                 aCol < a._cols &&
                 bCol < b._cols);

#ifdef HAVE_CBLAS_H
#  ifdef ISEP_ROW_MAJOR
    return cblas_ddot(a._rows,
                      a._data + aCol, a._cols,
                      b._data + bCol, b._cols);
#  else // !ISEP_ROW_MAJOR
    return cblas_ddot(a._rows,
                      a._data + aCol * a._rows, 1,
                      b._data + bCol * b._rows, 1);
#  endif
#else // !HAVE_CBLAS_H
    double result = 0.0;
    for (unsigned int i = 0; i < a._rows; i++)
        result += a(i, aCol) * b(i, bCol);
    return result;
#endif
}


double Matrix::dotRowRow(const Matrix &a, unsigned int aRow,
                         const Matrix &b, unsigned int bRow)
{
    debug_assert(a._cols == b._cols &&
                 aRow < a._rows &&
                 bRow < b._rows);

#ifdef HAVE_CBLAS_H
#  ifdef ISEP_ROW_MAJOR
    return cblas_ddot(a._cols,
                      a._data + aRow * a._cols, 1,
                      b._data + bRow * b._cols, 1);
#  else // !ISEP_ROW_MAJOR
    return cblas_ddot(a._cols,
                      a._data + aRow, a._rows,
                      b._data + bRow, b._rows);
#  endif
#else // !HAVE_CBLAS_H
    double result = 0.0;
    for (unsigned int i = 0; i < a._cols; i++)
        result += a(aRow, i) * b(bRow, i);
    return result;
#endif
}


Matrix Matrix::inverse() const
{
    if (_rows != _cols)
        throw std::runtime_error("Matrix is singular.");

    // Concatenate this matrix and an identity matrix. Then perform gaussian
    // elimination to determine the inverse of this matrix.
    Matrix foo(_rows, _cols * 2, generators::zero);
    for (unsigned int i = 0; i < _cols; i++) {
        foo.setColumn(i, nthColumn(i));
        foo(i, i + _cols) = 1.0;
    }
    foo.gaussElimination(true); // throws an exception if foo is singular.

    // Assemble the resulting matrix.
    Matrix result(_rows, _cols);
    for (unsigned int i = 0; i < _rows; i++)
        for (unsigned int j = 0; j < _cols; j++)
            result(i,j) = foo(i, j + _cols);

    return result;
}


Matrix Matrix::pseudoInverse() const
{
    return (this->transposed() * *this)
           .inverse()
           .multWithTransposedMatrix(*this);
}


Matrix Matrix::covarianceMatrix() const
{
    debug_assert(_cols > 1);

    Matrix foo(*this);

    ColVector foo_mean = foo.meanColumnVector();
    for (unsigned int i = 0; i < foo.rows(); i++)
        for (unsigned int j = 0; j < foo.cols(); j++)
            foo(i,j) -= foo_mean(i);

    return 1.0 / ((double)_cols - 1) * foo.multWithTransposedMatrix(foo);
}


const Matrix& Matrix::operator = (const Matrix& other)
{
    if (_rows != other._rows || _cols != other._cols) {
        delete[] _data;
        _rows = other._rows;
        _cols = other._cols;
        _data = new double[_rows * _cols];
    }
    memcpy(_data, other._data, _rows * _cols * sizeof(double));
    return *this;
}


bool Matrix::operator == (const Matrix& other) const
{
    if (_rows != other._rows || _cols != other._cols)
        return false;

    return (0 == memcmp(_data, other._data, _rows * _cols * sizeof(double)));
}


bool Matrix::operator != (const Matrix& other) const
{
    return (!(*this == other));
}


Matrix Matrix::operator * (const Matrix& other) const
{
    Matrix result(_rows, other._cols);
    this->multWithMatrix(other, &result);
    return result;
}


ColVector Matrix::operator * (const ColVector& cv) const
{
    debug_assert(_cols == cv.dim());

    ColVector result(_rows);

#ifdef HAVE_CBLAS_H
#  ifdef ISEP_ROW_MAJOR
    cblas_dgemv(CblasRowMajor, CblasNoTrans, _rows, _cols, 1.0,
                _data, _cols,
                cv._data, 1,
                0.0, result._data, 1);
#  else // !ISEP_ROW_MAJOR
    cblas_dgemv(CblasColMajor, CblasNoTrans, _rows, _cols, 1.0,
                _data, _rows,
                cv._data, 1,
                0.0, result._data, 1);
#  endif // ISEP_ROW_MAJOR
#else // !HAVE_CBLAS_H
    for (unsigned int i = 0; i < _rows; i++) {
        result(i) = 0;
        for (unsigned int j = 0; j < _cols; j++) {
            result(i) += this->at(i,j) * cv.at(j);
        }
    }
#endif

    return result;
}


Matrix Matrix::operator * (double s) const
{
    Matrix result(_rows, _cols);

#ifdef ISEP_ROW_MAJOR
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
#else
    for (unsigned int j = 0; j < _cols; j++) {
        for (unsigned int i = 0; i < _rows; i++) {
#endif
            result(i,j) = s * this->at(i,j);
        }
    }

    return result;
}


ColVector Matrix::meanColumnVector() const
{
    ColVector mv(_rows, generators::zero);
    const double f = 1 / (double)_cols;
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++)
            mv(i) += f * this->at(i,j);
    }
    return mv;
}


RowVector Matrix::meanRowVector() const
{
    RowVector mv(_cols, generators::zero);
    const double f = 1 / (double)_rows;
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++)
            mv(j) += f * this->at(i,j);
    }
    return mv;
}


ColVector Matrix::varianceRows() const
{
    ColVector ev = meanColumnVector();
    ColVector var(ev.dim(), generators::zero);
    const double f = 1 / (double)_cols;
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
            double tmp = this->at(i,j) - ev(i);
            var(i) += f * tmp * tmp;
        }
    }
    return var;
}


RowVector Matrix::varianceColumns() const
{
    RowVector ev = meanRowVector();
    RowVector var(ev.dim(), generators::zero);
    const double f = 1 / (double)_rows;
    for (unsigned int i = 0; i < _rows; i++) {
        for (unsigned int j = 0; j < _cols; j++) {
            double tmp = this->at(i,j) - ev(j);
            var(j) += f * tmp * tmp;
        }
    }
    return var;
}


void Matrix::dump(const std::string &fileName) const
{
    do {
        std::ofstream fos(fileName.c_str(),
            std::ios::out | std::ios::binary | std::ios_base::trunc);
        if (fos.fail())
            break;

        BinaryWriter bw(fos, BinaryWriter::LittleEndian);

        bw << uint32_t(2);
        bw << _rows;
        bw << _cols;
        if (fos.fail())
            break;

#ifdef ISEP_ROW_MAJOR
        bool ok = true;
        for (unsigned int j = 0; ok && j < _cols; ++j) {
            for (unsigned int i = 0; i < _rows; ++i) {
                if (bw.fail()) {
                    ok = false;
                    break;
                }
                bw << at(i, j);
            }
        }
        if (!ok) break;
#else
        if (bw.writeDoubles(_data, _rows * _cols) != _rows * _cols)
            break;
#endif

        // Everything's ok, so return at this point.
        return;
    } while (false);

    // Something went wrong.
    std::string msg("Error while writing matrix data to file ");
    msg.append(fileName);
    throw std::runtime_error(msg);
}


#ifdef _DEBUG
void Matrix::dumpRowMajor(const char* fileName, unsigned int prec) const
{
    std::ofstream os(fileName, std::ios_base::out | std::ios_base::trunc);
    if (os.fail())
        throw std::runtime_error("Unable to open file!");

    for (unsigned int j = 0; j < _cols; j++) {
        for (unsigned int i = 0; i < _rows; i++) {
            os << std::setprecision(prec) << this->at(i, j) << ' ';
        }
        os << std::endl;
    }

    os.close();
}
#endif


} // namespace linalg

} // namespace blissart
