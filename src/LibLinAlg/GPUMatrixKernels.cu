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

#include <cuda.h>
#include <blissart/linalg/common.h>


namespace blissart {


namespace linalg {


namespace gpu {


int blocksize = 4;


__global__ void MatrixAdd_d(const Elem *a, const Elem *b, Elem *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] + b[index];
}


void apply_add(const Elem* a, const Elem* b, Elem* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixAdd_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixSub_d(const Elem *a, const Elem *b, Elem *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] - b[index];
}


void apply_sub(const Elem* a, const Elem* b, Elem* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixSub_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixMul_d(const Elem *a, const Elem *b, Elem *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] * b[index];
}


void apply_mul(const Elem* a, const Elem* b, Elem* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixMul_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixDiv_d(const Elem *a, const Elem *b, Elem *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] / b[index];
}


void apply_div(const Elem* a, const Elem* b, Elem* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixDiv_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixPow_d(const Elem *a, const Elem b, Elem *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = pow(a[index], b);
}


void apply_pow(const Elem* a, const Elem b, Elem* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixPow_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void SetZero_d(Elem* a, int rows, int cols, 
                          int startRow, int startCol, int endRow, int endCol)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if (col >= startCol && col <= endCol && row >= startRow && row <= endRow)
        a[index] = 0.0f;
}


void set_to_zero(Elem* a, int rows, int cols, 
                 int startRow, int startCol, int endRow, int endCol)
{
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(cols / dimBlock.x + 1, rows / dimBlock.y + 1);
    SetZero_d<<<dimGrid, dimBlock>>>
        (a, rows, cols, startRow, startCol, endRow, endCol);
    cudaThreadSynchronize();
}


__global__ void Floor_d(Elem* a, const Elem floor, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows && a[index] < floor) 
        a[index] = floor;
}


void apply_floor(Elem* a, const Elem floor, int rows, int cols)
{
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(cols / dimBlock.x + 1, rows / dimBlock.y + 1);
    Floor_d<<<dimGrid, dimBlock>>>(a, floor, rows, cols);
    cudaThreadSynchronize();
}


} // namespace gpu


} // namespace linalg


} // namespace blissart
#endif
