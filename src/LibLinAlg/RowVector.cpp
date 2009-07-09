//
// $Id: RowVector.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <stdexcept>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/Matrix.h>

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


RowVector::RowVector(const std::string& fileName) : Vector(fileName)
{
    if (!checkOrientationFlag()) {
        std::string msg("Expected row vector in file ");
        msg.append(fileName);
        throw std::runtime_error(msg);
    }
}


double RowVector::operator * (const ColVector& vc) const
{
    debug_assert(vc._dim == _dim);

#ifdef HAVE_CBLAS_H
    return cblas_ddot(_dim, _data, 1, vc._data, 1);
#else
    double result = 0;
    for (unsigned int i = 0; i < _dim; i++)
        result += this->at(i) * vc.at(i);

    return result;
#endif
}


RowVector RowVector::operator * (const Matrix& m) const
{
    debug_assert(_dim == m.rows());

    // TODO: Make use of ATLAS if possible.

    RowVector result(m.cols());
    for (unsigned int i = 0; i < m.cols(); i++) {
        result(i) = 0;
        for (unsigned int j = 0; j < _dim; j++) {
            result(i) += this->at(j) * m.at(j,i);
        }
    }
    return result;
}


RowVector RowVector::operator * (double s) const
{
    RowVector result(*this);
    for (unsigned int i = 0; i < _dim; i++)
        result(i) *= s;
    return result;
}


ColVector RowVector::transposed() const
{
    return ColVector(_dim, _data);
}


RowVector& RowVector::operator = (const RowVector& other)
{
    return static_cast<RowVector&>(Vector::operator=(other));
}


RowVector RowVector::operator + (const RowVector& other) const
{
    debug_assert(other._dim == _dim);

    RowVector result(*this);
    result.add(other);
    return result;
}


RowVector RowVector::operator - (const RowVector& other) const
{
    debug_assert(other._dim == _dim);

    RowVector result(*this);
    result.sub(other);
    return result;
}


void RowVector::operator += (const RowVector& other)
{
    debug_assert(other._dim == _dim);

    for (unsigned int i = 0; i < _dim; i++)
        _data[i] += other._data[i];
}


void RowVector::operator -= (const RowVector& other)
{
    debug_assert(other._dim == _dim);

    for (unsigned int i = 0; i < _dim; i++)
        _data[i] -= other._data[i];
}


} // namespace linalg

} // namespace blissart
