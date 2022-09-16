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
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/generators/generators.h>
#include <stdexcept>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#ifdef HAVE_CBLAS_H
extern "C" {
#include <cblas.h>
}
#endif



#ifdef BLISSART_SINGLE_PREC
#    define CBLAS_SCAL cblas_sscal
#    define CBLAS_GER  cblas_sger
#else
#    define CBLAS_SCAL cblas_dscal
#    define CBLAS_GER  cblas_dger
#endif


namespace blissart {

namespace linalg {


ColVector::ColVector(const std::string& fileName) : Vector(fileName)
{
    if (!checkOrientationFlag()) {
        std::string msg("Expected column vector in file ");
        msg.append(fileName);
        throw std::runtime_error(msg);
    }
}


Matrix ColVector::operator * (const RowVector& rv) const
{
#ifdef HAVE_CBLAS_H
    Elem *result_data = new Elem[_dim * rv._dim];
    CBLAS_SCAL(_dim * rv._dim, 0.0, result_data, 1);

    #  ifdef ISEP_ROW_MAJOR
    CBLAS_GER(CblasRowMajor, _dim, rv._dim, 1.0, _data, 1, rv._data, 1,
               result_data, rv._dim);
#  else // !ISEP_ROW_MAJOR
    CBLAS_GER(CblasColMajor, _dim, rv._dim, 1.0, _data, 1, rv._data, 1,
               result_data, _dim);
#  endif // ISEP_ROW_MAJOR

    return Matrix(_dim, rv._dim, result_data, true /* useRawPointer */);
#else // !HAVE_CBLAS_H
    Matrix result(_dim, rv._dim);

    for (unsigned int i = 0; i < _dim; i++)
        for (unsigned int j = 0; j < rv._dim; j++)
            result.setAt(i, j, _data[i] * rv._data[j]);

    return result;
#endif
}


RowVector ColVector::transposed() const
{
    return RowVector(_dim, _data);
}


ColVector ColVector::operator * (Elem s) const
{
    ColVector result(*this);
    for (unsigned int i = 0; i < _dim; i++)
        result(i) *= s;
    return result;
}


//ColVector& ColVector::operator = (const ColVector& other)
//{
    //return static_cast<ColVector&>(Vector::operator=(other));
//}
ColVector &ColVector::operator = (const ColVector& other)
{
    debug_assert(other._dim == _dim);
//    return static_cast<RowVector&>(Vector::operator=(other));
    ColVector result(*this);
    result.equal(other);
    return *this;
}


ColVector ColVector::operator + (const ColVector& other) const
{
    debug_assert(other._dim == _dim);

    ColVector result(*this);
    result.add(other);
    return result;
}


ColVector ColVector::operator - (const ColVector& other) const
{
    debug_assert(other._dim == _dim);

    ColVector result(*this);
    result.sub(other);
    return result;
}


void ColVector::operator += (const ColVector& other)
{
    debug_assert(other._dim == _dim);

    for (unsigned int i = 0; i < _dim; i++)
        _data[i] += other._data[i];
}


void ColVector::operator -= (const ColVector& other)
{
    debug_assert(other._dim == _dim);

    for (unsigned int i = 0; i < _dim; i++)
        _data[i] -= other._data[i];
}


} // namespace linalg

} // namespace blissart
