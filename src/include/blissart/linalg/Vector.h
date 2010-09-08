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


#ifndef __BLISSART_LINALG_VECTOR_H__
#define __BLISSART_LINALG_VECTOR_H__


#include <common.h>
#include <types.h>
#include <iostream>
#include <iomanip>


namespace blissart {

namespace linalg {


// Forward declaration
class Matrix;


/**
 * \addtogroup linalg
 * @{
 */

/**
 * Abstract base class for real row- and column-vectors.
 */
class LibLinAlg_API Vector 
{
public:
    /** 
     * @name Constructors
     * @{
     */


    /**
     * Constructs a Vector with the given dimension.
     * @param   dim         the desired dimension
     */
    explicit Vector(unsigned int dim);


    /**
     * Constructs a Vector from another given vector.
     * @param   other       the other vectors which dimension and data will be
     *                      used for initilization of the new object
     */
    Vector(const Vector& other);


    /**
     * Constructs a Vector with the given dimension and from the provided data.
     * @param   dim         the desired dimension
     * @param   data        a double array with the initialization data
     */
    Vector(unsigned int dim, const double* data);


    /**
     * Constructs a Vector with the given dimension and fills it's entries with
     * the results from the specified callback function.
     * @param   dim         the desired dimension
     * @param   generator   a function pointer to the generator
     */
    Vector(unsigned int dim, double (*generator)(unsigned int i));


    /**
     * Reads a Vector from a binary file.
     * @param   fileName    the name of the source file
     * @throw               std::runtime_error
     */
    explicit Vector(const std::string& fileName);
    

    /* @} */


    /**
     * Destroys the Vector and frees all memory that was allocated for data
     * representation.
     */
    virtual ~Vector(void);


    /** 
     * @name Vector entries
     * Functions that return or modify vector entries.
     * @{
     */


    /**
     * Returns a reference to the Vector entry at the specified position.
     * The entry cannot be modified by the caller.
     * @param   i           the position
     * @return              the vector element at position i.
     */
    inline const double& at(unsigned int i) const;


    /**
     * Returns a reference to the Vector entry at the specified position.
     * The entry can be modified by the caller.
     * @param   i           the position
     * @return              the vector element at position i.
     */
    inline double& at(unsigned int i);


    /**
     * Sets the specified entry to the given value.
     * @param   i           the position
     * @param   d           the new value
     */
    inline void setAt(unsigned int i, double d);

    
    /**
     * Returns a reference to the element at the specified position and hence
     * isn't declared as const.
     * @param   i           the position
     * @return              a reference to the vector element at position i.
     */
    inline double& operator () (unsigned int i);


    /**
     * Returns a "const" reference to the element at the specified position.
     * Thus the entry cannot be modified by the caller.
     * @param   i           the position
     * @return              a reference to the vector element at position i.
     */
    inline const double& operator () (unsigned int i) const;


    /* @} */


    /** 
     * @name Vector operations
     * @{
     */


    /**
     * Scales this vector according to the given scalar value.
     * @param   s           the scalar value
     */
    void scale(double s);

    
    /**
     * Compares this vectors' dimension and all of its entries with another
     * vector.
     * @param   other       the other Vector to which this object
     *                      should be compared
     * @return              true if and only if dimension as well as all
     *                      entries are the same for both vectors and
     *                      either both are row- or both are colvectors.
     */
    bool operator == (const Vector& other) const;


    /**
     * Compares this vectors' dimension and all of its entries with another
     * vector.
     * @param   other       the other Vector to which this object
     *                      should be compared
     * @return              true if and only if either dimension or 1+ entries
     *                      of both vectors differ or if they are not of
     *                      the same type row-/colvector.
     */
    bool operator != (const Vector& other) const;


    /**
     * Copies all entries from another vector.
     * @param   other       another Vector with the same dimensions.
     * @return              a reference to this object.
     */
    virtual Vector& operator = (const Vector& other);


    /**
     * Computes the inner product of this and the given vector.
     * @param   other       a Vector.
     * @return              a double value which holds the inner product.
     */
    double operator * (const Vector& other) const;


    /**
     * Scales this Vector according to the given scalar value.
     * @param   s           a real value
     */
    void operator *= (double s);


    /**
     * Normalizes this vector.
     */
    void normalize();


    /**
     * Shifts the vector by 1 position to the "right".
     */
    void shiftRight();


    /**
     * Shifts the vector by 1 position to the "right".
     */
    void shiftLeft();


    /**
     * Adds another vector to this one.
     * @param   other       another vector object with the same dimension
     */
    void add(const Vector& other);


    /**
     * Subtracts another vector from this one.
     * @param   other       another vector object with the same dimension
     */
    void sub(const Vector& other);


    /**
     * Returns the maximum value in this vector.
     * @param   abs       whether to use absolute value for comparison or not
     */
    double maximum(bool abs = false) const;


    /**
     * Returns the minimum value in this vector.
     * @param   abs       whether to use absolute value for comparison or not
     */
    double minimum(bool abs = false) const;


    /**
     * Fills this vector with random values and normalizes to unit length
     * iff possible, i.e. at least one of the vector's components is != 0
     */
    void randomize();


    /**
     * Computes the angle between the two given vectors.
     * @param   a           the first Vector
     * @param   b           the second Vector
     * @return              the angle (RAD) between a and b
     */
    static double angle(const Vector& a, const Vector& b);

    
    /* @} */


    /** 
     * @name Basic vector properties
     * @{
     */


    /**
     * @return              true if and only if this vector is a row vector
     */
    virtual bool isRowVector() const = 0;

    
    /**
     * @return              true if and only if this vector is a column vector
     */
    virtual bool isColVector() const { return !isRowVector(); }


    /**
     * Returns the dimension of this Vector.
     * @return              an unsigned int representing this vector's dimension.
     */
    inline unsigned int dim() const { return _dim; }


    /**
     * Calculates the length of this vector.
     * @return              a double value representing the length of this vector.
     */
    double length() const;


    /* @} */


    /** 
     * @name Vector I/O
     * @{
     */


    /**
     * Dumps the Vector in binary format to a file.
     * @param   fileName    the name of the destination file
     * @throw               std::runtime_error
     */
    void dump(const std::string& fileName) const;


    /**
     * Outputs a Vector to the given stream.
     * @param   os          the output stream
     * @param   v           the Vector whose output is desired.
     * @return              a reference to the given stream.
     */
    friend std::ostream& operator << (std::ostream& os, const Vector& v);


    /* @} */


protected:
    // Forbid default constructor (actually this isn't neccessary because of the
    // 1+ abstract methods above).
    Vector();


    // Checks the orientation flag when constructing a Vector object from a file.
    // This method needs to be called by RowVector and ColVector subclasses
    // since isRowVector() is pure virtual.
    virtual bool checkOrientationFlag();


    // The orientation flag which is set when constructing a Vector object 
    // from a file.
    uint32_t _orientationFlag;


    /**
     * Computes the inner product of two given vectors with the same dimension.
     * @param   a           the first Vector
     * @param   b           the second Vector
     * @return              the component-wise product of a and b
     */
    static double inner_prod(const Vector& a, const Vector& b);


    unsigned int _dim;
    double *     _data;
};


/**
 * @}
 */


inline std::ostream& operator << (std::ostream& os, const Vector& v)
{
    os << "(";
    for (unsigned int i = 0; i < v.dim(); i++) {
        os  << std::setiosflags(std::ios::fixed)
            << std::setprecision(2)
            << v.at(i);
        if (i < v.dim()-1)
            os << ", ";
    }
    os << ")";
    return os;
}


// Inlines


const double& Vector::at(unsigned int i) const
{
    debug_assert(i < _dim);
    return _data[i];
}


double& Vector::at(unsigned int i)
{
    debug_assert(i < _dim);
    return _data[i];
}


void Vector::setAt(unsigned int i, double d)
{
    debug_assert(i < _dim);
    _data[i] = d;
}


double& Vector::operator () (unsigned int i)
{
    debug_assert(i < _dim);
    return _data[i];
}


const double& Vector::operator () (unsigned int i) const
{
    debug_assert(i < _dim);
    return _data[i];
}


} // namespace linalg

} // namespace blissart


#endif // __BLISSART_LINALG_VECTOR_H__
