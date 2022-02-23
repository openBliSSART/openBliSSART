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


#include <blissart/linalg/Vector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/BinaryReader.h>
#include <blissart/BinaryWriter.h>

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
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
#    define CBLAS_DOT  cblas_sdot
#    define CBLAS_NRM2 cblas_snrm2
#    define CBLAS_SCAL cblas_sscal
#    define CBLAS_AXPY cblas_saxpy
#else
#    define CBLAS_DOT  cblas_ddot
#    define CBLAS_NRM2 cblas_dnrm2
#    define CBLAS_SCAL cblas_dscal
#    define CBLAS_AXPY cblas_daxpy
#endif


namespace blissart {

namespace linalg {


Vector::Vector(unsigned int dim) :
    _dim(dim),
    _data(new Elem[dim])
{
    debug_assert(dim > 0);
}


Vector::Vector(const Vector& other) :
    _dim(other._dim),
    _data(new Elem[other._dim])
{
    memcpy(_data, other._data, _dim * sizeof(Elem));
}


Vector::Vector(unsigned int dim, const Elem* data) :
    _dim(dim),
    _data(new Elem[dim])
{
    debug_assert(dim > 0);
    memcpy(_data, data, _dim * sizeof(Elem));
}


Vector::Vector(unsigned int dim, Elem (*generator)(unsigned int i)) :
    _dim(dim),
    _data(new Elem[dim])
{
    debug_assert(dim > 0);
    // Set all entries according to the provided generator
    for (unsigned int i = 0; i < _dim; i++)
        _data[i] = generator(i);
}


Vector::Vector(const std::string &fileName) :
    _data(0)
{
    do {
        std::ifstream fis(fileName.c_str(), std::ios::in | std::ios::binary);
        if (fis.fail())
            break;

        BinaryReader br(fis, BinaryReader::LittleEndian);

        br >> _orientationFlag;
        if (_orientationFlag > 1)
            break;
        // orientation itself checked later

        br >> _dim;
        if (_dim <= 0)
            break;

        _data = new Elem[_dim];
#ifdef BLISSART_SINGLE_PREC
        if (br.readFloats(_data, _dim) != _dim)
#else
        if (br.readDoubles(_data, _dim) != _dim)
#endif
            break;

        // Everything's ok, so return at this point.
        return;
    } while (false);

    // Something went wrong.
    if (_data)
        delete _data;
    std::string msg("Error while reading vector data from file ");
    msg.append(fileName);
    throw std::runtime_error(msg);
}


bool Vector::checkOrientationFlag() {
    return (this->isRowVector() && _orientationFlag == 0) || 
           (this->isColVector() && _orientationFlag == 1);
}


Vector::~Vector(void)
{
    delete[] _data;
}


void Vector::dump(const std::string& fileName) const
{
    do {
        std::ofstream fos(fileName.c_str(),
            std::ios::out | std::ios::binary | std::ios_base::trunc);
        if (fos.fail())
            break;

        BinaryWriter bw(fos, BinaryWriter::LittleEndian);

        bw << (this->isRowVector() ? uint32_t(0) : uint32_t(1));
        bw << _dim;
        if (fos.fail())
            break;

#ifdef BLISSART_SINGLE_PREC
        if (bw.writeFloats(_data, _dim) != _dim)
#else
        if (bw.writeDoubles(_data, _dim) != _dim)
#endif
            break;

        // Everything's ok, so return at this point.
        return;
    } while (false);

    // Something went wrong.
    std::string msg("Error while writing vector data to file ");
    msg.append(fileName);
    throw std::runtime_error(msg);
}


bool Vector::operator == (const Vector& other) const
{
    return (
        this->isRowVector() == other.isRowVector() &&
        (0 == memcmp(_data, other._data, _dim * sizeof(Elem)))
        );
}


bool Vector::operator != (const Vector& other) const
{
    return !(*this == other);
}


Vector& Vector::operator = (const Vector& other)
{
    if (_dim != other._dim) {
        delete[] _data;
        _dim = other._dim;
        _data = new Elem[_dim];
    }
    memcpy(_data, other._data, _dim * sizeof(Elem));
    return *this;
}


Elem Vector::inner_prod(const Vector& a, const Vector& b)
{
    debug_assert(a._dim == b._dim);

#ifdef HAVE_CBLAS_H
    return CBLAS_DOT(a._dim, a._data, 1, b._data, 1);
#else
    Elem result = 0;
    for (unsigned int i = 0; i < a._dim; i++)
        result += a._data[i] * b._data[i];
    return result;
#endif
}


Elem Vector::length() const
{
#ifdef HAVE_CBLAS_H
    return CBLAS_NRM2(_dim, _data, 1);
#else
    return sqrt(inner_prod(*this, *this));
#endif
}


Elem Vector::angle(const Vector& a, const Vector& b)
{
    return acos(inner_prod(a, b) / (a.length() * b.length()) );
}


void Vector::scale(Elem s)
{
#ifdef HAVE_CBLAS_H
    CBLAS_SCAL(_dim, s, _data, 1);
#else
    for (unsigned int i = 0; i < _dim; i++)
        _data[i] *= s;
#endif
}


void Vector::normalize()
{
    Elem length = this->length();
    if (length != 0)
        scale(1.0 / length);
}


void Vector::shiftLeft()
{
    for (unsigned int i = 0; i < _dim - 1; ++i) {
        _data[i] = _data[i + 1];
    }
    _data[_dim - 1] = 0.0;
}


void Vector::shiftRight()
{
    for (int j = (int) _dim - 1; j >= 1; --j) {
        _data[j] = _data[j - 1];
    }
    _data[0] = 0.0;
}


void Vector::add(const Vector& other)
{
    debug_assert(other._dim == _dim);

#ifdef HAVE_CBLAS_H
    CBLAS_AXPY(_dim, 1.0, other._data, 1, _data, 1);
#else
    Elem* const maxData = _data + _dim;
    for (Elem *d = _data, *dd = other._data; d < maxData; d++, dd++)
        *d += *dd;
#endif
}


void Vector::sub(const Vector& other)
{
    debug_assert(other._dim == _dim);

#ifdef HAVE_CBLAS_H
    CBLAS_AXPY(_dim, -1.0, other._data, 1, _data, 1);
#else
    Elem* const maxData = _data + _dim;
    for (Elem *d = _data, *dd = other._data; d < maxData; d++, dd++)
        *d -= *dd;
#endif
}


Elem Vector::maximum(bool abs) const
{
    debug_assert(_dim > 0);
    Elem max = _data[0];
    if (abs) {
        for (unsigned int i = 1; i < _dim; ++i) {
            if (fabs(_data[i]) > fabs(max))
                max = _data[i];
        }
    } else {
        for (unsigned int i = 1; i < _dim; ++i) {
            if (_data[i] > max)
                max = _data[i];
        }
    }
    return max;
}


Elem Vector::minimum(bool abs) const
{
    debug_assert(_dim > 0);
    Elem min = _data[0];
    if (abs) {
        for (unsigned int i = 1; i < _dim; ++i) {
            if (fabs(_data[i]) < fabs(min))
                min = _data[i];
        }
    } else {
        for (unsigned int i = 1; i < _dim; ++i) {
            if (_data[i] < min)
                min = _data[i];
        }
    }
    return min;
}


void Vector::randomize()
{
    for (unsigned int i = 0; i < _dim; i++)
        _data[i] = (Elem)rand() / (Elem)RAND_MAX;
    normalize();
}


void Vector::operator *= (Elem s)
{
    scale(s);
}


} // namespace linalg

} // namespace blissart
