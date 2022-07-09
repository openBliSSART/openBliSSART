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

#undef HAVE_CUDA

#ifdef HAVE_CUDA
#include <blissart/linalg/GPUMatrix.h>
#include <blissart/linalg/GPUUtil.h>
#include <blissart/linalg/GPUMatrixKernels.h>
#include <blissart/linalg/generators/generators.h>
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <cublas_v2.h>


#ifdef BLISSART_SINGLE_PREC
#   define CUBLAS_GEMM cublasSgemm
#   define CUBLAS_SCAL cublasSscal
#else
#   define CUBLAS_GEMM cublasDgemm
#   define CUBLAS_SCAL cublasDscal
#endif


namespace blissart {


namespace linalg {


GPUMatrix::GPUMatrix(const Matrix& hostMatrix) :
    _rows(hostMatrix.rows()),
    _cols(hostMatrix.cols()),
    _data(0)
{
    initDeviceMemory();

    // Copy data to the device.
    cublasStatus_t cublasStat = cublasSetMatrix(
        hostMatrix.rows(), hostMatrix.cols(), 
        sizeof(*hostMatrix._data), 
        hostMatrix._data, hostMatrix.rows(), 
        _data, _rows
    );
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Data transfer to GPU failed!");
    }
}


GPUMatrix::GPUMatrix(unsigned int rows, unsigned int cols) :
    _rows(rows),
    _cols(cols)
{
    initDeviceMemory();
}


GPUMatrix::~GPUMatrix()
{
    if (_data != 0)
        cudaFree(_data);
}


void GPUMatrix::initDeviceMemory()
{
    // Initialize CUBLAS if necessary.
    GPUStart();

    // Allocate device memory to fit the given matrix dimensions.
    cudaError_t cudaStat = cudaMalloc((void**) &_data, 
        _rows * _cols * sizeof(*_data));
    if (cudaStat != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory!");
    }
}


void GPUMatrix::multWithMatrix(const GPUMatrix& other, GPUMatrix* target) const
{
    multWithMatrix(other, target, 
        false, false, 
        this->_rows, this->_cols, other._cols, 
        0, 0, 0, 0, 0, 0);
}


void GPUMatrix::multWithTransposedMatrix(const GPUMatrix& other, GPUMatrix* target) const
{
    multWithMatrix(other, target, 
        false, true, 
        this->_rows, this->_cols, other._rows, 
        0, 0, 0, 0, 0, 0);
}


void GPUMatrix::transposedMultWithMatrix(const GPUMatrix& other, GPUMatrix* target) const
{
    multWithMatrix(other, target, 
        true, false, 
        this->_cols, this->_rows, other._cols, 
        0, 0, 0, 0, 0, 0);
}


void GPUMatrix::multWithMatrix(const GPUMatrix& other, GPUMatrix* target,
    bool transpose, bool transposeOther,
    unsigned int m, unsigned int k, unsigned int n,
    unsigned int rowOffset, unsigned int colOffset,
    unsigned int rowOffsetOther, unsigned int colOffsetOther,
    unsigned int rowOffsetTarget, unsigned int colOffsetTarget) const
{
    // We might allocate these on the device later, to get the last bit of performance.
    const Elem alpha = 1.0;
    const Elem beta  = 0.0;
    cublasStatus_t rv = CUBLAS_GEMM(
        _cublasHandle,
        transpose      ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeOther ? CUBLAS_OP_T : CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        _data + colOffset * _rows + rowOffset,
        this->_rows,    // lda
        other._data + colOffsetOther * other._rows + rowOffsetOther,
        other._rows,    // ldb
        &beta,
        target->_data + colOffsetTarget * target->_rows + rowOffsetTarget,
        target->_rows   // ldc
    );
    if (rv != CUBLAS_STATUS_SUCCESS) 
        // TODO: define exception class with cublas return codes?
        throw std::runtime_error("CUBLAS error in Dgemm");
}


void GPUMatrix::add(const GPUMatrix &other)
{
    this->add(other, this);
}


void GPUMatrix::add(const GPUMatrix &other, GPUMatrix* target)
{
    assert(this->_rows == other._rows && other._rows == target->_rows &&
           this->_cols == other._cols && other._cols == target->_cols);
    gpu::apply_add(this->_data, other._data, target->_data, this->_rows, this->_cols);
}


void GPUMatrix::sub(const GPUMatrix &other)
{
    this->sub(other, this);
}


void GPUMatrix::sub(const GPUMatrix &other, GPUMatrix* target)
{
    // TODO: check dimensions
    gpu::apply_sub(this->_data, other._data, target->_data, this->_rows, this->_cols);
}


void GPUMatrix::elementWiseMult(const GPUMatrix &other, GPUMatrix* target)
{
    gpu::apply_mul(this->_data, other._data, target->_data, this->_rows, this->_cols);
}


void GPUMatrix::elementWiseDiv(const GPUMatrix &other, GPUMatrix* target)
{
    gpu::apply_div(this->_data, other._data, target->_data, this->_rows, this->_cols);
}


void GPUMatrix::elementWisePow(const Elem exp, GPUMatrix* target)
{
    gpu::apply_pow(this->_data, exp, target->_data, this->_rows, this->_cols);
}


void GPUMatrix::zero()
{
    gpu::set_to_zero(this->_data, this->_rows, this->_cols, 
        0, 0, this->_rows - 1, this->_cols - 1);
}


void GPUMatrix::zero(unsigned int startRow, unsigned int startCol,
                     unsigned int endRow,   unsigned int endCol)
{
    gpu::set_to_zero(this->_data, this->_rows, this->_cols, 
        startRow, startCol, endRow, endCol);
}


void GPUMatrix::floor(Elem value)
{
    gpu::apply_floor(this->_data, value, this->_rows, this->_cols);
}


void GPUMatrix::getMatrix(Matrix* target)
{
    assert(target->_rows == _rows && target->_cols == _cols);
    cublasStatus_t cublasStat = cublasGetMatrix(
        _rows, _cols,
        sizeof(*target->_data),
        _data, _rows,
        target->_data, target->_rows
    );
    if (cublasStat != CUBLAS_STATUS_SUCCESS) 
        // TODO: define exception class with cublas return codes?
        throw std::runtime_error("CUBLAS error in GetMatrix");
}


void GPUMatrix::scale(const Elem alpha, unsigned int startCol, 
                      unsigned int endCol)
{
    assert(endCol >= startCol && startCol < _cols && endCol < _cols);
    cublasStatus_t cublasStat = CUBLAS_SCAL(
        _cublasHandle, 
        (endCol - startCol + 1) * _rows,   // number of elements to scale
        &alpha,
        _data + startCol * _rows,          // include col offset
        1            // increment of 1 for column-wise scaling
    );
}


void GPUMatrix::scale(const Elem alpha)
{
    scale(alpha, 0, this->_cols - 1);
}


void GPUMatrix::rowSums(GPUMatrix* sums)
{
    Matrix x(_cols, 1, generators::unity);
    GPUMatrix xgpu(x);
    this->multWithMatrix(xgpu, sums);
}


void GPUMatrix::colSums(GPUMatrix* sums)
{
    Matrix x(1, _rows, generators::unity);
    GPUMatrix xgpu(x);
    xgpu.multWithMatrix(*this, sums);
}


} // namespace linalg


} // namespace blissart
#endif
