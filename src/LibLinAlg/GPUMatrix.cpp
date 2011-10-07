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


#include <blissart/linalg/GPUMatrix.h>
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>


namespace blissart {


namespace linalg {


bool            GPUMatrix::_cublasInitialized = false;
cublasHandle_t  GPUMatrix::_cublasHandle;


GPUMatrix::GPUMatrix(Matrix& hostMatrix) : // Matrix(hostMatrix)
    Matrix(hostMatrix.rows(), hostMatrix.cols())
{
    cudaError_t cudaStat;
    cublasStatus_t cublasStat;

    // Initialize CUBLAS if necessary.
    if (!_cublasInitialized) {
        GPUStart();
    }

    // Allocate device memory to fit the host matrix.
    cudaStat = cudaMalloc((void**) &_dataDev, 
        hostMatrix.rows() * hostMatrix.cols() * sizeof(*_dataDev));
    if (cudaStat != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory!");
    }

    // Copy data to the device.
    cublasStat = cublasSetMatrix(
        hostMatrix.rows(), hostMatrix.cols(), 
        sizeof(*hostMatrix.dataPtr()), 
        hostMatrix.dataPtr(), hostMatrix.rows(), 
        _dataDev, rows()
    );
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Data transfer to GPU failed!");
    }
}


void GPUMatrix::multWithMatrix(const GPUMatrix& other, GPUMatrix* target) const
{
    multWithMatrix(other, target, false, false, this->rows(), this->cols(), 
        other.cols(), 0, 0, 0, 0, 0, 0);
}


void GPUMatrix::multWithMatrix(const GPUMatrix& other, GPUMatrix* target,
    bool transpose, bool transposeOther,
    unsigned int m, unsigned int k, unsigned int n,
    unsigned int rowOffset, unsigned int colOffset,
    unsigned int rowOffsetOther, unsigned int colOffsetOther,
    unsigned int rowOffsetTarget, unsigned int colOffsetTarget) const
{
    // We might allocate these on the device later, to get the last bit of performance.
    const double alpha = 1.0;
    const double beta  = 0.0;
    cublasStatus_t rv = cublasDgemm(
        _cublasHandle,
        transpose      ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeOther ? CUBLAS_OP_T : CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        _dataDev + colOffset * this->rows() + rowOffset,
        this->rows(),    // lda
        other._dataDev + colOffsetOther * other.rows() + rowOffsetOther,
        other.rows(),    // ldb
        &beta,
        target->_dataDev + colOffsetTarget * target->rows() + rowOffsetTarget,
        target->rows()   // ldc
    );
    if (rv != CUBLAS_STATUS_SUCCESS) 
        // TODO: define exception class with cublas return codes?
        throw std::runtime_error("CUBLAS error in Dgemm");
}


GPUMatrix::~GPUMatrix()
{
    cudaFree(_dataDev);
}


void GPUMatrix::GPUStart()
{
    cublasStatus_t cublasStat;
    cublasStat = cublasCreate(&_cublasHandle);
    if (cublasStat != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Could not initialize CUBLAS!");
    _cublasInitialized = true;
}


void GPUMatrix::GPUStop()
{
    if (_cublasInitialized)
        cublasDestroy(_cublasHandle);
}


void GPUMatrix::getMatrix(Matrix* target)
{
    assert(target->rows() == rows() && target->cols() == cols());
    cublasStatus_t cublasStat = cublasGetMatrix(
        rows(), cols(),
        sizeof(*target->dataPtr()),
        _dataDev, rows(),
        target->dataPtr(), target->rows()
    );
}


} // namespace linalg


} // namespace blissart
