#include <stdio.h>
#include "include/fluid_sim.cuh"
#include "include/gradient.cuh"
#include "include/fluid_utils.cuh"

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

__global__ void thread_idx_ND(size_t *buffer,
    const size_t nRows,
    const size_t nCols,
    const size_t nDims)
{
    // Example of row, col access into 2D arrays
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    // Conditional guard since the grid may operate
    // out-of-bounds
    if (y < nRows && x < nCols)
        buffer[(y * nCols + x ) * nDims + z] = z;
}

void test_ND_dim()
{
    const size_t nRows = 20;
    const size_t nCols = 20;
    const size_t nDims = 2; // 2-dimension vector
    size_t buffer[nRows][nCols][nDims];
    size_t *d_buffer;

    // Method of calculating multi-dimensional kernel calls
    dim3 dimBlock(32,32); // This is the maximum as per CUDA 2.x
    dim3 dimGrid( // Method of calculating the number of blocks to use
        (nCols + dimBlock.x - 1) / dimBlock.x,
        (nRows + dimBlock.y - 1) / dimBlock.y,
        nDims);

    memset(buffer, 0, sizeof(buffer));
    cudaMalloc(&d_buffer,sizeof(size_t) * nRows * nCols * 2);
    cudaMemcpy(d_buffer, buffer, sizeof(buffer), cudaMemcpyHostToDevice);
    thread_idx_ND<<<dimGrid,dimBlock>>>(d_buffer, nRows, nCols, nDims);
    cudaMemcpy(buffer, d_buffer, sizeof(buffer), cudaMemcpyDeviceToHost);
    //cudaError err = cudaDeviceSynchronize();
    //printf("%d\n",err);
    for (size_t idx = 0; idx < nRows; idx++)
    {
        for (size_t jdx = 0; jdx < nCols; jdx++)
        {
            printf("(%d, %d) ", buffer[idx][jdx][0], buffer[idx][jdx][1]);
        }
        printf("\n");
    }
    cudaFree(d_buffer);
}

/// @brief 
/// @param dim_x 
/// @param dim_y 
/// @param data 
/// @return 


void test_advect()
{
    size_t idx, jdx;
    MatrixDim dim = {5,5,2};

    dim3 dimBlock(32,32); // This is the maximum as per CUDA 2.x
    dim3 dimGrid2( // Method of calculating the number of blocks to use
        (dim.x + dimBlock.x - 1) / dimBlock.x,
        (dim.y + dimBlock.y - 1) / dimBlock.y);
#if 0
    dim3 dimGrid3( // Method of calculating the number of blocks to use
        (dim.x + dimBlock.x - 1) / dimBlock.x,
        (dim.y + dimBlock.y - 1) / dimBlock.y,
        2);
#endif
    float pdata[dim.y][dim.x][2] =
    {
        {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}},
        {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}},
        {{0.0,0.0},{0.0,0.0},{0.0,1.0},{0.0,0.0},{0.0,0.0}},
        {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}},
        {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}}
    };
    float *ddata;
    cudaMalloc(&ddata, sizeof(pdata));

    cudaMemcpy(ddata, pdata, sizeof(pdata), cudaMemcpyHostToDevice);
    kernel_advect<<<dimGrid2, dimBlock>>>(dim, ddata, ddata, 5, 0.1);
    cudaMemcpy(pdata, ddata, sizeof(pdata), cudaMemcpyDeviceToHost);
    for ( idx = 0; idx < dim.y; idx++)
    {
        for ( jdx = 0; jdx < dim.x; jdx++)
        {
            printf("(%f, %f) ", pdata[idx][jdx][0], pdata[idx][jdx][1]);
        }
        printf("\n");
    }

    unsigned int bgr[dim.y][dim.x];
    memset(bgr, 0, sizeof(bgr));
    unsigned int *dbgr;
    cudaMalloc(&dbgr, sizeof(bgr));

    cudaMemcpy(dbgr, bgr, sizeof(bgr), cudaMemcpyHostToDevice);
    kernel_gradient<<<dimGrid2, dimBlock>>>(ddata, dbgr, dim);
    cudaMemcpy(bgr, dbgr, sizeof(bgr), cudaMemcpyDeviceToHost);

    for ( idx = 0; idx < dim.y; idx++)
    {
        for ( jdx = 0; jdx < dim.x; jdx++)
        {
            printf("0x%X ", bgr[idx][jdx]);
        }
        printf("\n");
    }

    cudaFree(ddata);
    cudaFree(dbgr);
}

void test_bilinear_interpolation()
{
    const MatrixDim dim = { 2, 2, 2};
    float data[dim.x][dim.y][dim.vl] =
    {
        {{-1.0, 1.0}, {1.0, 1.0}},
        {{-1.0, -1.0}, {1.0, -1.0}}
    };

    float vx = 10.0, vy = 10.0;
    bilinear_interpolation(0.05,0.5,(float *)data,dim,&vx,&vy);
    printf("%f, %f\n",vx, vy);
}

int main()
{
    //test_2D_dim();
    //test_ND_dim();
    //test_advect();
    test_bilinear_interpolation();
    return 0;
}