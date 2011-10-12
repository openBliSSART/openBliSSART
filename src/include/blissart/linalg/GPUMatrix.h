//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2011, Alexander Lehmann <lehmanna@in.tum.de>
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


#ifndef __BLISSART_LINALG_GPUMATRIX_H__
#define __BLISSART_LINALG_GPUMATRIX_H__


#ifdef ISEP_ROW_MAJOR
# error CUDA does not support row-major layout
#endif


#include <blissart/linalg/Matrix.h>
//#include <cublas_v2.h>


namespace blissart {


namespace linalg {


namespace gpu {
// TODO document
void apply_add(const double* a, const double* b, double* c, int m, int n);
void apply_sub(const double* a, const double* b, double* c, int m, int n);
void apply_mul(const double* a, const double* b, double* c, int m, int n);
void apply_div(const double* a, const double* b, double* c, int m, int n);
void apply_pow(const double* a, const double b, double* c, int m, int n);
void set_to_zero(double* a, int m, int n, int startRow, int startCol, int endRow, int endCol);
}


/**
 * Represents a matrix on the GPU.
 */
class GPUMatrix //: protected Matrix
{
public:
    // Reserves space on the GPU, but does not copy any data.
    // This is only useful if this serves as a target, e.g. for a matrix multiplication on the GPU.
    GPUMatrix(unsigned int rows, unsigned int cols);
    // Reserves space on the GPU and copies data from host matrix.
    GPUMatrix(const Matrix& hostMatrix);
    virtual ~GPUMatrix();

    void multWithMatrix(const GPUMatrix& other, GPUMatrix* target) const;
    void multWithMatrix(const GPUMatrix& other, GPUMatrix* target,
        bool transpose, bool transposeOther,
        unsigned int m, unsigned int k, unsigned int n,
        unsigned int rowOffset, unsigned int colOffset,
        unsigned int rowOffsetOther, unsigned int colOffsetOther,
        unsigned int rowOffsetTarget, unsigned int colOffsetTarget) const;

    void add(const GPUMatrix& other);
    void add(const GPUMatrix& other, GPUMatrix* target);
    void sub(const GPUMatrix& other);
    void sub(const GPUMatrix& other, GPUMatrix* target);
    void elementWiseMult(const GPUMatrix& other, GPUMatrix* target);
    void elementWiseDiv(const GPUMatrix& other, GPUMatrix* target);
    void elementWisePow(const double exp, GPUMatrix* target);

    void getMatrix(Matrix* target);
    
    inline unsigned int rows() const { return _rows; }
    inline unsigned int cols() const { return _cols; }

    /**
     * Resets all matrix entries to zero.
     */
    void zero();
    
    
    /**
     * Sets the specified submatrix to zero.
     */
    void zero(unsigned int startRow, unsigned int startCol,
              unsigned int endRow,   unsigned int endCol);
              


private:
    void initDeviceMemory();
    unsigned int _rows;
    unsigned int _cols;
    double *     _data;
};


} // namespace linalg


} // namespace blissart


#endif // __BLISSART_LINALG_GPUMATRIX_H__
