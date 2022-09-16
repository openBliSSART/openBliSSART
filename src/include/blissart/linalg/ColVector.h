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

#ifndef __BLISSART_LINALG_COLVECTOR_H__
#define __BLISSART_LINALG_COLVECTOR_H__


#include <blissart/linalg/common.h>
#include <blissart/linalg/Vector.h>


namespace blissart {

namespace linalg {


// Forward declaration
class RowVector;


/**
 * \defgroup linalg Linear Algebra (LibLinAlg)
 * \addtogroup linalg
 * @{
 */

/**
 * Representation of real column-vectors.
 */
class LibLinAlg_API ColVector : public Vector
{
public:
    /**
     * Constructs a ColVector with the given dimension.
     * @param   dim         the desired dimension
     */
    explicit ColVector(unsigned int dim) : Vector(dim) {}


    /**
     * Constructs a ColVector from another ColVector. The new ColVector then has
     * the same dimensions and the same data as the other.
     * @param   o           another ColVector
     */
    ColVector(const ColVector& o) : Vector(o) {}


    /**
     * Constructs a ColVector with the specified dimension and initial data.
     * @param   dim         the desired dimension
     * @param   data        a Elem array of the initial elements
     */
    ColVector(unsigned int dim, const Elem* data) : Vector(dim, data) {}


    /**
     * Constructs a ColVector with the given dimension and fills it's entries
     * with the results from the specified callback function.
     * @param   dim         the desired dimension
     * @param   generator   a function pointer to the generator
     */
    ColVector(unsigned int dim, Elem (*generator)(unsigned int i))
        : Vector(dim, generator) {}


    /**
     * Reads a ColVector from a binary file.
     * @param   fileName    the name of the source file
     * @throw               std::runtime_error
     */
    explicit ColVector(const std::string& fileName);


    /**
     * Sets this vectors dimension to the dimension of the given vector and
     * copies all elements. Memory allocation is rearranged only if neccessary.
     * @param   other       another ColVector
     * @return              a reference to this object
     */
    //virtual ColVector& operator = (const ColVector& other);
    ColVector& operator = (const ColVector& other);



    /**
     * Determines whether this object is a RowVector.
     * @return              true iff this object is a RowVector.
     */
    virtual bool isRowVector() const { return false; }


    /**
     * Multiplies this ColVector by the given RowVector.
     * @param   rv          a RowVector
     * @returns             a Matrix
     */
    Matrix operator * (const RowVector& rv) const;


    /**
     * Returns a transposed copy of this vector.
     * @return              the transpose of this ColVector as RowVector
     */
    RowVector transposed() const;


    /**
     * Multiplies this vector with the given scalar value.
     * @param   s           the scalar value
     * @return              an appropriately scaled ColVector
     */
    ColVector operator * (Elem s) const;


    /**
     * Computes the sum of this and another given column vector.
     * @param   other       another column vector
     * @return              the corresponding difference as ColVector
     */
    ColVector operator + (const ColVector& other) const;


    /**
     * Computes the difference of this and another given column vector.
     * @param   other       another column vector
     * @return              the corresponding difference as ColVector
     */
    ColVector operator - (const ColVector& other) const;


    /**
     * Adds another ColVector to this ColVector.
     * @param   other       another ColVector
     */
    void operator += (const ColVector& other);


    /**
     * Subtracts another ColVector from this ColVector.
     * @param   other       another ColVector
     */
    void operator -= (const ColVector& other);


    /**
     * Multiplies a ColVector with the given scalar value.
     * @param   s           the scalar value
     * @param   cv          the ColVector
     * @return              an appropriately scaled ColVector
     */
    friend ColVector operator * (Elem s, const ColVector& cv);


    // We want to be friends with classes RowVector and Matrix.
    friend class RowVector;
    friend class Matrix;
};


/**
 * @}
 */


// Inlines


inline ColVector operator * (Elem s, const ColVector& cv)
{
    return cv * s;
}


} // namespace linalg

} // namespace blissart


#endif // __BLISSART_LINALG_COLVECTOR_H__
