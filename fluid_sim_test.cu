#include <stdio.h>
#include "include/fluid_sim.cuh"

__global__ void thread_idx_2D(size_t *buffer, const size_t nCols)
{
    // Example of row, col access into 2D arrays
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    buffer[row * nCols + col] = col;
}

void test_2D_dim()
{
    const size_t nRows = 512;
    const size_t nCols = 512;
    size_t buffer[nRows][nCols];
    size_t *d_buffer;

    // Method of calculating multi-dimensional kernel calls
    dim3 dimBlock(32, 32); // This is the maximum as per CUDA 2.x
    dim3 dimGrid( // Method of calculating the number of blocks to use
        (nCols + dimBlock.x - 1) / dimBlock.x,
        (nRows + dimBlock.y - 1) / dimBlock.y);

    memset(buffer, 0, sizeof(buffer));
    cudaMalloc(&d_buffer,sizeof(size_t) * nRows *nCols);
    cudaMemcpy(d_buffer, buffer, sizeof(buffer), cudaMemcpyHostToDevice);
    thread_idx_2D<<<dimGrid,dimBlock>>>(d_buffer, nCols);
    cudaMemcpy(buffer, d_buffer, sizeof(buffer), cudaMemcpyDeviceToHost);
    for (size_t idx = 0; idx < nRows; idx++)
    {
        for (size_t jdx = 0; jdx < nCols; jdx++)
            printf("%d ", buffer[idx][jdx]);
        printf("\n");
    }
    cudaFree(d_buffer);
}

#if 0
__global__ void thread_idx_3D(size_t *buffer, const size_t nCols)
{
    // Example of row, col access into 2D arrays
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    buffer[row * nCols + col] = col;
}
#endif

/// @brief 
/// @param dim_x 
/// @param dim_y 
/// @param data 
/// @return 
__global__ void kernel_advect(
    const size_t dim_x,
    const size_t dim_y,
    float* data
)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t rdx = 1;
    const float timestep = 0.1;
    const float dissipation = 0.999;

    float dx_new, dy_new;

    advect(dim_x,dim_y,x,y,
        rdx,timestep,dissipation,
        data,data,&dx_new, &dy_new);
}

int main()
{
    test_2D_dim();
    return 0;
}