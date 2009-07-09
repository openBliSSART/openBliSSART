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

#ifndef __BLISSART_LINALG_ROWVECTOR_H__
#define __BLISSART_LINALG_ROWVECTOR_H__


#include <blissart/linalg/Vector.h>


namespace blissart {

namespace linalg {


// Forward declaration
class ColVector;


/**
 * Representation of real row-vectors.
 */
class LibLinAlg_API RowVector : public Vector
{
public:
    /**
     * Constructs a RowVector with the given dimension.
     * @param   dim         the desired dimension
     */
    explicit RowVector(unsigned int dim) : Vector(dim) {}


    /**
     * Constructs a RowVector from another RowVector. The new RowVector then has
     * the same dimensions and the same data as the other.
     * @param   o           another RowVector
     */
    RowVector(const RowVector& o) : Vector(o) {}


    /**
     * Constructs a RowVector with the specified dimension and initial data.
     * @param   dim         the desired dimension
     * @param   data        a double array of the initial elements
     */
    RowVector(unsigned int dim, const double* data) : Vector(dim, data) {}


    /**
     * Constructs a RowVector with the given dimension and fills it's entries
     * with the results from the specified callback function.
     * @param   dim         the desired dimension
     * @param   generator   a function pointer to the generator
     */
    RowVector(unsigned int dim, double (*generator)(unsigned int i))
        : Vector(dim, generator) {}


    /**
     * Reads a RowVector from a binary file.
     * @param   fileName    the name of the source file
     * @throw               std::runtime_error
     */
    explicit RowVector(const std::string& fileName);


    /**
     * Sets this vectors dimension to the dimension of the given vector and
     * copies all elements. Memory allocation is rearranged only if neccessary.
     * @param   other       another RowVector
     * @return              a reference to this object
    */
    virtual RowVector& operator = (const RowVector& other);
    
    
    /**
     * Determines whether this object is a RowVector.
     * @return              true if and only if this object is a RowVector.
     */
    virtual bool isRowVector() const { return true; }


    /**
     * Returns a transposed copy of this vector.
     * @return              the transpose of this RowVector as ColVector
     */
    ColVector transposed() const;


    /**
     * Multiplies this RowVector by the specified ColVector, i.e. calculates the
     * inner product of the two vectors.
     * @param   cv          a ColVector
     * @returns             the scalar result
     */
    double operator * (const ColVector& cv) const;


    /**
     * Multiplies this RowVector by the specified Matrix.
     * @param   m           a Matrix
     * @return              a RowVector
     */
    RowVector operator * (const Matrix& m) const;


    /**
     * Multiplies this vector with the given scalar value.
     * @param   s           the scalar value
     * @return              an appropriately scaled RowVector
     */
    RowVector operator * (double s) const;


    /**
     * Computes the sum of this and another given RowVector.
     * @param   other       another RowVector
     * @return              the corresponding difference as RowVector
     */
    RowVector operator + (const RowVector& other) const;


    /**
     * Computes the difference of this and another given RowVector.
     * @param   other       another RowVector
     * @return              the corresponding difference as RowVector
     */
    RowVector operator - (const RowVector& other) const;


    /**
     * Adds another RowVector to this RowVector.
     * @param   other       another RowVector
     */ 
    void operator += (const RowVector& other);


    /**
     * Subtracts another RowVector from this RowVector.
     * @param   other       another RowVector
     */ 
    void operator -= (const RowVector& other);


    /**
     * Multiplies a RowVector with the given scalar value.
     * @param   s           the scalar value
     * @param   rv          a RowVector
     * @return              an appropriately scaled RowVector
     */
    friend RowVector operator * (double s, const RowVector& rv);


    // We want to be friends with the ColVector class
    friend class ColVector;
};


// Inlines


inline RowVector operator * (double s, const RowVector& rv)
{
    return rv * s;
}


} // namespace linalg

} // namespace blissart


#endif // __BLISSART_LINALG_ROWVECTOR_H__
