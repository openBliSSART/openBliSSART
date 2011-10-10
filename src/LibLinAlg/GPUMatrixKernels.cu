#include <cuda.h>

namespace blissart {
namespace linalg {
namespace gpu {


int blocksize = 4;


__global__ void MatrixAdd_d(const double *a, const double *b, double *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] + b[index];
}


void apply_add(const double* a, const double* b, double* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixAdd_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixSub_d(const double *a, const double *b, double *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] - b[index];
}


void apply_sub(const double* a, const double* b, double* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixSub_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixMul_d(const double *a, const double *b, double *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] * b[index];
}


void apply_mul(const double* a, const double* b, double* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixMul_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixDiv_d(const double *a, const double *b, double *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = a[index] / b[index];
}


void apply_div(const double* a, const double* b, double* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixDiv_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void MatrixPow_d(const double *a, const double b, double *c, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if(col < cols && row < rows) 
        c[index] = pow(a[index], b);
}


void apply_pow(const double* a, const double b, double* c, int m, int n) {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(n / dimBlock.x + 1, m / dimBlock.y + 1);
    MatrixPow_d<<<dimGrid, dimBlock>>>(a, b, c, m, n);
    cudaThreadSynchronize();
}


__global__ void SetZero_d(double* a, int rows, int cols, 
                          int startRow, int startCol, int endRow, int endCol)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * rows + row;
    if (col >= startCol && col <= endCol && row >= startRow && row <= endRow)
        a[index] = 0.0f;
}


void set_to_zero(double* a, int rows, int cols, 
                 int startRow, int startCol, int endRow, int endCol)
{
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(cols / dimBlock.x + 1, rows / dimBlock.y + 1);
    SetZero_d<<<dimGrid, dimBlock>>>
        (a, rows, cols, startRow, startCol, endRow, endCol);
    cudaThreadSynchronize();
}


}
}
}
