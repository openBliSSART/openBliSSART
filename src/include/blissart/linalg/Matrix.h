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


#ifndef __BLISSART_LINALG_MATRIX_H__
#define __BLISSART_LINALG_MATRIX_H__


#include <common.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <blissart/linalg/common.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>

namespace blissart {


/**
 * Classes that implement matrix and vector operations.
 */
namespace linalg {


// Forward declaration
class ColVector;
class RowVector;


// Uncomment the following line if you want the matrices to have a row-major
// storage layout (default is column-major):
//#define ISEP_ROW_MAJOR


/**
 * \addtogroup linalg
 * @{
 */

/**
 * Representation of real matrices.
 */
class LibLinAlg_API Matrix
{
    friend class GPUMatrix;


public:
    /**
     * List of eigenpairs, i.e. a list of pairs of which the first item
     * is the eigenvalue and the second is the associated eigenvector.
     */
    typedef std::vector<std::pair<Elem, ColVector> > EigenPairs;


    /**
     * Type of a generator function for matrices.
     */
    typedef Elem (*GeneratorFunction)(unsigned int row, unsigned int column);


    /**
     * @name Constructors
     * @{
     */


    /**
     * Constructs a Matrix from another Matrix.
     * @param other         another Matrix whose entries should be copied.
     */
    Matrix(const Matrix& other);


    /**
     * Constructs a Matrix with the specified dimensions. Note that the elements
     * of this newly created matrix will *not* be initialized. \see generators
     * @param  rows         number of rows
     * @param  cols         number of columns
     */
    Matrix(unsigned int rows, unsigned int cols);


    /**
     * Constructs a Matrix with the specified dimensions from given data.  The
     * data is assumed to be in row-major order (if you don't know what this
     * means, then don't use this!)
     * @param  rows            number of rows
     * @param  cols            number of columns
     * @param  data            data array
     * @param  useRawPointer   whether to use the given pointer or copy the data
     */
    Matrix(unsigned int rows, unsigned int cols, const Elem* data,
           bool useRawPointer = false);


    /**
     * Constructs a Matrix with the specified dimensions and fills its values
     * through the provided generator function.
     */
    Matrix(unsigned int rows, unsigned int cols,
           Elem (*generator)(unsigned int i, unsigned int j));


    /**
     * Reads a Matrix from a binary file.
     * @param   fileName    the name of the source file
     * @throw               std::runtime_error
     */
    explicit Matrix(const std::string& fileName);


    /* @} */


    /** 
     * Returns a vector of Matrix objects read from a binary file.
     * The vector length will be one if the file contains a matrix,
     * and greater or equal to one if the file contains a tensor.
     */
    static std::vector<Matrix*> arrayFromFile(const std::string& file);


    /**
     * Destroys the Matrix. Frees all memory used for data representation.
     */
    virtual ~Matrix();


    /**
     * @name Matrix entries
     * Functions that return or modify one or more matrix entries.
     * @{
     */


    /**
     * Returns a reference to the matrix entry at the specified position and
     * thus isn't marked as 'const'.
     * @param  i            row index
     * @param  j            column index
     * @return              the matrix entry at position (i,j)
     */
    inline Elem& at(unsigned int i, unsigned int j);


    /**
     * Returns a "const" reference to the matrix entry at the specified
     * position, thus the entry cannot be modified by the caller.
     * @param  i            row index
     * @param  j            column index
     * @return              reference to the matrix entry at position (i,j)
     */
    inline const Elem& at(unsigned int i, unsigned int j) const;


    /**
     * Returns a reference to the matrix entry at the specified position and
     * thus isn't marked as 'const'.
     * @param  i            row index
     * @param  j            column index
     * @return              reference to the matrix entry at position (i,j)
     */
    inline Elem& operator () (unsigned int i, unsigned int j);


    /**
     * Returns a "const" reference to the matrix entry at the specified
     * position, thus the entry cannot be modified by the caller.
     * @param  i            row index
     * @param  j            column index
     * @return              reference to the matrix entry at position (i,j)
     */
    inline const Elem& operator () (unsigned int i, unsigned int j) const;


    /**
     * Sets the matrix entry at the specified position (i,j) to the given value.
     * @param   i           row index
     * @param   j           column index
     * @param   d           the new value
     */
    inline void setAt(unsigned int i, unsigned int j, Elem d);


    /**
     * Returns the nth column of this matrix.
     * @param   n           the number of the column
     * @return              a ColVector representing the entries of the
     *                      specified column
     */
    ColVector nthColumn(unsigned int n) const;


    /**
     * Returns the nth row of this matrix.
     * @param    n          the number of the row
     * @return              a RowVector representing the entries of the
     *                      specified row
     */
    RowVector nthRow(unsigned int n) const;


    /**
     * Copies the contents of this matrix' nth row into the specified Elem
     * array.
     * @param   n           the number of the row
     * @param   da          a Elem array
     */
    void nthRow2DoubleArray(unsigned int n, Elem* da) const;


    /**
     * Returns a copy of this matrix up to and including the given row. The rows
     * are numbered as usual, i.e. the first starting with 0.
     * @param   row         the last row which should be included
     * @return              a corresponding new matrix
     */
    Matrix upToAndIncludingRow(unsigned int row) const;


    /**
     * Sets the specified column to the values of the given column vector.
     * @param   col         the number of the column
     * @param   cv          a column vector
     */
    void setColumn(unsigned int col, const ColVector& cv);


    /**
     * Sets the specified row to the values of the given row vector.
     * @param   row         the number of the row
     * @param   rv          a row vector
     */
    void setRow(unsigned int row, const RowVector& rv);


    /**
     * Copy the specified source row of the given matrix to the specified
     * destination row of this matrix.
     * @param   dstRow      which row to copy to
     * @param   srcRow      which row to copy from
     * @param   other       another matrix with the same column dimension
     */
    void copyRow(unsigned int dstRow, unsigned int srcRow, const Matrix& other);


    /* @} */


    /**
     * @name Basic matrix properties
     * @{
     */


    /**
     * Returns the number of rows for this Matrix.
     * @return              the # of rows
     */
    inline unsigned int rows() const;


    /**
     * Returns the number of columns for this Matrix.
     * @return              the # of columns
     */
    inline unsigned int cols() const;


    /**
     * Returns whether this matrix is symmetric or not.
     * @return              true if and only if this matrix is symmetrical
     */
    bool isSymmetric() const;


    /**
     * Determines whether this matrix is quadratic.
     * @return              true if and only if this matrix is quadratic,
     *                      i.e. it's dimensions are equal
     */
    bool isQuadratic() const;


    /**
     * Returns the Frobenius norm of this matrix.
     */
    Elem frobeniusNorm() const;


    /* @} */


    /**
     * @name Matrix operations
     * @{
     */


    /**
     * Performs matrix multiplication and stores the result in the given target
     * matrix. This function is provided for convenience only.
     * @param   other       a Matrix
     * @param   target      a pointer to a Matrix that will hold the results
     */
    void multWithMatrix(const Matrix& other, Matrix* target) const;


    /**
     * A versatile function for matrix multiplication, supporting
     * submatrices and transposing.
     * @param   other       the Matrix to multiply with
     * @param   target      a pointer to a Matrix holding the result
     * @param   transpose   whether to transpose this matrix in multiplication
     * @param   transposeOther whether to transpose the other matrix
     * @param   m           the desired number of rows of the result
     * @param   k           the desired dimension of the multiplication
     * @param   n           the desired number of columns of the result
     * @param   rowOffset   the row offset for this Matrix
     * @param   colOffset   the column offset for this Matrix
     * @param   rowOffsetOther   the row offset for the other Matrix
     * @param   colOffsetOther   the column offset for the other Matrix
     * @param   rowOffsetTarget   the row offset for the target Matrix
     * @param   colOffsetTarget   the column offset for the target Matrix
     */
    void multWithMatrix(const Matrix& other, Matrix* target,
        bool transpose, bool transposeOther,
        unsigned int m, unsigned int k, unsigned int n,
        unsigned int rowOffset, unsigned int colOffset,
        unsigned int rowOffsetOther, unsigned int colOffsetOther,
        unsigned int rowOffsetTarget, unsigned int colOffsetTarget) const;


    /**
     * Multiplies this matrix by the given matrix' transpose.
     * This function is a wrapper for multWithMatrix and is provided for
     * convenience only.
     * @param   other       a Matrix
     * @return              the product of this matrix and the given
     *                      matrix' transpose
     */
    Matrix multWithTransposedMatrix(const Matrix& other) const;


    /**
     * Multiplies this matrix by the given matrix' transpose and stores the
     * result in the given target matrix.
     * Note that this operation does not use extra memory for transposing the
     * matrix.
     * This function is a wrapper for multWithMatrix and is provided for
     * convenience only.
     * @param   other       a Matrix
     * @param   target      a pointer to a Matrix that will hold the results
     */
    void multWithTransposedMatrix(const Matrix& other, Matrix* target) const;


    /**
     * Multiplies the transposed of this matrix by the given matrix and stores 
     * the result in the given target matrix.
     * Note that this operation does not use extra memory for transposing the
     * matrix.
     * This function is a wrapper for multWithMatrix and is provided for
     * convenience only.
     * @param   other       a Matrix
     * @param   target      a pointer to a Matrix that will hold the results
     */
    void transposedMultWithMatrix(const Matrix& other, Matrix* target) const;


    /**
     * TODO
     */
    void apply(Elem (*func) (Elem, Elem), const Matrix& other);
    void apply(Elem (*func) (Elem, Elem), const Matrix& other, Matrix* target) const;
    void apply(Elem (*func) (Elem, Elem), Elem other);
    void apply(Elem (*func) (Elem, Elem), Elem other, Matrix* target) const;

    inline static Elem mul(Elem a, Elem b);
    inline static Elem div(Elem a, Elem b);


    /**
     * Ensures that all matrix elements are less or equal to the given floor
     * value.
     */
    void floor(double f);


    /**
     * Performs element-wise division of this matrix elements by another given
     * matrix elements and stores the result in the given target matrix.
     * @param   other       a Matrix
     * @param   target      a pointer to a Matrix that will hold the results
     */
    void elementWiseDivision(const Matrix& other, Matrix* target) const;


    /**
     * Computes the Hadamard product (element-wise multiplication) of this 
     * matrix with another matrix and stores the result in the given target
     * which may be identical to the original matrix (in-place operation).
     * @param   other       a Matrix
     * @param   target      a pointer to a Matrix that will hold the results
     */
    void elementWiseMultiplication(const Matrix& other, Matrix* target) const;


    /**
     * TODO
     */
    void pow(double exponent);


    /**
     * Stores the transpose of the matrix in the given Matrix.
     * @param   target      a pointer to a Matrix where the result will be stored
     */
    void transpose(Matrix* target) const;


    /**
     * Returns a copy of the transpose of this matrix.
     * @return              a matrix object
     */
    Matrix transposed() const;


    /**
     * Shifts the columns of the matrix to the right by one column, 
     * introducing a zero-column on the left.
     */
    void shiftColumnsRight();


    /**
     * Shifts the columns of the matrix to the left by one column, 
     * introducing a zero-column on the right.
     */
    void shiftColumnsLeft();


    /**
     * Performs element-wise in-place matrix addition.
     * @param   other       a Matrix whose elements should be added to
     *                      this matrix
     */
    void add(const Matrix& other);


    /**
     * Performs element-wise in-place matrix subtraction.
     * @param   other       a Matrix whose elements should be subtracted from
     *                      this matrix
     */
    void sub(const Matrix& other);


    /**
     * Resets all matrix entries to zero.
     */
    void zero();
    
    
    /**
     * Transforms the matrix into row echelon form by using Gauss elimination.
     * @param   reducedRowEchelonForm
     *                      whether to compute (reduced) row echelon form
     * @return              the number of rows which have been swapped due to
     *                      selection of pivot elements
     * @throw               a std::runtime_error in case of an under-determined
     *                      Matrix
     */
    unsigned int gaussElimination(bool reducedRowEchelonForm = false);


    /**
     * Solves equations of the form A * x = b.
     * @param   m           a Matrix whose elements from the LSE
     * @param   b           a pointer to a ColVector, i.e. the right-hand side vector
     * @param   target      a pointer to a ColVector which will hold the results
     */
    static void linearSolve(const Matrix& m, const ColVector& b, ColVector* target);


    /**
     * Calculates the determinant of this matrix (only for square matrices).
     * @param   triangularHint  since the calculation of triangular matrices'
     *                          determinants is somewhat faster this
     *                          boolean parameter helps suppress the
     *                          needed Gaussian Elimination for bigger matrices
     * @return                  the determinant
     */
    double determinant(const bool triangularHint = false);


    /**
     * Calculates the trace of the matrix.
     * @return              the trace of the matrix, i.e. the sum of all of it's
     *                      diagonal elements
     */
    double trace() const;


    /**
     * Calculates the eigenvalues and unit-length eigenvectors of this matrix.
     * This function uses power iteration (also known as power method).
     * @param   maxNrOfEigenPairs   the maximum number of eigenpairs that should
     *                              be computed (all if 0)
     * @param   maxIter             the maximum number of iterations
     * @param   prec                the desired precision
     * @return                      a list of eigenpairs sorted by their
     *                              absolute values in descending order
     *                              (dominant eigenvalues)
     */
    EigenPairs eigenPairs(unsigned int maxNrOfEigenPairs = 0,
                          unsigned int maxIter = 10,
                          double prec = 1e-10) const;


    /**
     * Eliminates all negative elements of the matrix by setting them to 0.
     */
    void eliminateNegativeElements();


    /**
     * Returns the sum of all elements of the given column.
     * @param   column              the number of the column
     * @return                      the sum of the column's elements
     */
    Elem colSum(unsigned int column) const;


    /**
     * Returns the sum of all elements of the given row.
     * @param   row                 the number of the row
     * @return                      the sum of the row's elements
     */
    Elem rowSum(unsigned int row) const;

    /**
     * Returns the sum of the elements of the given row in the given column 
     * range.
     * @param   row                 the number of the row
     * @param   col1                start column
     * @param   col2                end column
     * @return                      the sum of the row's elements
     */
    Elem rowSum(unsigned int row, unsigned int col1, unsigned int col2) 
        const;


    /**
     * Computes the dot-product of the given matrices' columns.
     * @param   a                   a Matrix
     * @param   aCol                the column number for a
     * @param   b                   another Matrix
     * @param   bCol                the column number for b
     * @return                      the dot-product of the columns
     */
    static Elem dotColCol(const Matrix &a, unsigned int aCol,
                            const Matrix &b, unsigned int bCol);


    /**
     * Computes the dot-product of the given matrices' rows.
     * @param   a                   a Matrix
     * @param   aRow                the row number for a
     * @param   b                   another Matrix
     * @param   bRow                the row number for b
     * @return                      the dot-product of the rows
     */
    static Elem dotRowRow(const Matrix &a, unsigned int aRow,
                            const Matrix &b, unsigned int bRow);


    /**
     * Computes and returns the inverse of this matrix by gaussian elimination.
     * @return                      the inverse of this matrix
     * @throw                       std::runtime_error in case of a singular
     *                              matrix
     */
    Matrix inverse() const;


    /**
     * Computes and returns the Moore-Penrose pseudo-inverse of this matrix.
     * @return                      the pseudo-inverse of this matrix
     */
    Matrix pseudoInverse() const;


    /**
     * Computes and returns the covariance matrix of this matrix.
     * @return                      the covariance matrix of this matrix
     */
    Matrix covarianceMatrix() const;


    /**
     * Copies the entries from another matrix object and adjusts dimensions if
     * neccessary.
     * @param   other       the Matrix to copy
     * @return              a reference to this matrix
     */
    const Matrix& operator = (const Matrix& other);


    /**
     * Compares the matrix to another matrix.
     * @param   other       a Matrix to compare to
     * @return              true iff the matrices have the same dimension and
     *                      all matrix entries are equal.
     */
    bool operator == (const Matrix& other) const;


    /**
     * Compares the matrix to another matrix.
     * @param   other       a Matrix to compare to
     * @return              true iff the matrices have different dimension or
     *                      different matrix entries.
     */
    bool operator != (const Matrix& other) const;


    /**
     * Multiplies this matrix by the given one.
     * @param   other       another Matrix
     * @return              the resulting Matrix
     */
    Matrix operator * (const Matrix& other) const;


    /**
     * Multiplies this matrix by the given vector.
     * @param   cv          a ColVector
     * @return              the resulting ColVector
     */
    ColVector operator * (const ColVector& cv) const;


    /**
     * Multiplies this matrix by the given scalar.
     * @param   s           a scalar value
     * @return              a matrix where all elements equal this
     *                      matrix' elements multiplied by s
     */
    Matrix operator * (Elem s) const;


    /**
     * Multiplies a Matrix with the given scalar.
     * @param   s           a scalar value
     * @param   m           a Matrix
     * @return              a matrix where all elements equal this
     *                      matrix' elements multiplied by s
     */
    friend Matrix operator * (Elem s, const Matrix& m);


    /**
     * Computes the arithmetic mean of all column vectors of this matrix.
     * @return              the mean column vector
     */
    ColVector meanColumnVector() const;


    /**
     * Computes the arithmetic mean of all row vectors of this matrix.
     * @return              the mean row vector
     */
    RowVector meanRowVector() const;


    /**
     * Computes the variances of all rows of this matrix.
     * @return              a column vector with the corresponding variances
     */
    ColVector varianceRows() const;


    /**
     * Computes the variances of all columns of this matrix.
     * @return              a row vector with the corresponding variances
     */
    RowVector varianceColumns() const;


    /**
     * Compute the expected value of this matrix.
     * Note that T must support the += operator.
     * @param   iv          the initial value (this is neccessary because not
     *                      all classes have a default constructor)
     * @param   info        an arbitrary pointer at your disposal
     * @param   callback    a function pointer to a callback function that will
     *                      be called for every single column of this matrix
     *                      while carrying along the info pointer
     * @return              the return type depends on the explicit template
     *                      call to this function
     */
    template<class T>
    inline T expectedValue(const T& iv, void* info,
                    T (*callback)(const ColVector& cv, void* info)) const;


    /* @} */


    /**
     * @name Matrix I/O
     * @{
     */


    /**
     * Writes the Matrix object in binary form to the given file.
     * @param   fileName    the name of the destination file
     * @throw               std::runtime_error
     */
    void dump(const std::string& fileName) const;


    /**
     * Writes a vector of Matrix objects to binary ("tensor") file.
     * @param   fileName    the name of the destination file
     * @throw               std::runtime_error
     */
    static void arrayToFile(const std::vector<const Matrix*> mv, const std::string& file);


    /**
     * Outputs a matrix in human-readable form to the given stream.
     * @param   os          an output stream
     * @param   m           the matrix to output
     * @return              a reference to the given stream
     */
    friend std::ostream& operator << (std::ostream& os, const Matrix& m);


    /* @} */


#ifdef _DEBUG
    /**
     * Write the contents of this matrix into a text file with the given name in
     * row major order.
     * @param   fileName    the name of the output file
     * @param   prec        the desired precision
     * @throw               a std::runtime_error iff the output file couldn't
     *                      be opened
     */
    void dumpRowMajor(const char* fileName, unsigned int prec = 5) const;
#endif


protected:
    const Elem *dataPtr() const;
    Elem *dataPtr();

private:
    unsigned int _rows;
    unsigned int _cols;
    Elem *     _data;
};


/**
 * @}
 */
/*

template<class T>
T Matrix::expectedValue(const T& iv, void* info,
                        T (*callback)(const ColVector& cv, void* info)) const
{
    T result = iv;
    const double f = 1.0 / _cols;
		for (unsigned int i = 0; i < _cols; i++)     
		  result += f * callback(this->nthColumn(i), info);
    return result;
}
*/

//
// Inlines
//


inline Elem *Matrix::dataPtr()
{
    return _data;
}


inline const Elem *Matrix::dataPtr() const
{
    return _data;
}


inline std::ostream& operator << (std::ostream& os, const Matrix& m)
{
    unsigned int i, j;
    for (i = 0; i < m._rows; i++) {
        for (j = 0; j < m._cols; j++) {
            os  << std::setw(9)
                << std::setiosflags(std::ios::fixed)
                << std::setprecision(3)
                << m.at(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}


inline Matrix operator * (Elem s, const Matrix& m)
{
    return m * s;
}


const Elem& Matrix::at(unsigned int i, unsigned int j) const
{
    debug_assert(i < _rows && j < _cols);
#ifdef ISEP_ROW_MAJOR
    return _data[i * _cols + j];
#else
    return _data[j * _rows + i];
#endif
}


Elem& Matrix::at(unsigned int i, unsigned int j)
{
    debug_assert(i < _rows && j < _cols);
#ifdef ISEP_ROW_MAJOR
    return _data[i * _cols + j];
#else
    return _data[j * _rows + i];
#endif
}


Elem& Matrix::operator()(unsigned int i, unsigned int j)
{
    debug_assert(i < _rows && j < _cols);
#ifdef ISEP_ROW_MAJOR
    return _data[i * _cols + j];
#else
    return _data[j * _rows + i];
#endif
}


const Elem& Matrix::operator()(unsigned int i, unsigned int j) const
{
    debug_assert(i < _rows && j < _cols);
#ifdef ISEP_ROW_MAJOR
    return _data[i * _cols + j];
#else
    return _data[j * _rows + i];
#endif
}


void Matrix::setAt(unsigned int i, unsigned int j, Elem d)
{
    debug_assert(i < _rows && j < _cols);
#ifdef ISEP_ROW_MAJOR
    _data[i * _cols + j] = d;
#else
    _data[j * _rows + i] = d;
#endif
}


unsigned int Matrix::rows() const
{
    return _rows;
}


unsigned int Matrix::cols() const
{
    return _cols;
}


Elem Matrix::mul(Elem a, Elem b)
{
    return a * b;
}


Elem Matrix::div(Elem a, Elem b)
{
    return a / b;
}


} // namespace linalg

} // namespace blissart


#endif // __BLISSART_LINALG_MATRIX_H__
