#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "include/gradient.cuh"
#include "include/fluid_sim.cuh"
#include "include/fluid_utils.cuh"

static float rand_norm_scalar()
{
    float retval = (float) rand() / (float) RAND_MAX + 1.0;
    retval = retval * ((rand() % 2 > 0) ? 1.0 : -1.0);
    return retval;
}

static __host__ void initialize_p_field(float *data, const MatrixDim dim)
{
    const size_t radius = 300;
    const size_t x_lower = dim.x/2 - radius;
    const size_t x_upper = dim.x/2 + radius;
    const size_t y_lower = dim.y/2 - radius;
    const size_t y_upper = dim.y/2 + radius;

    for (size_t y = y_lower; y < y_upper; y++)
    {
        for (size_t x = x_lower; x < x_upper; x++)
        {
#if 0
            data[matrix_index(x,y,dim,0)] = 2.0 * rand_norm_scalar();
            data[matrix_index(x,y,dim,1)] = 2.0;// * rand_norm_scalar();
#else
            data[matrix_index(x,y,dim,0)] = rand_norm_scalar();
            data[matrix_index(x,y,dim,1)] = rand_norm_scalar();
#endif
        }
    }
}

static __host__ void initialize_bgr_field(unsigned int *data, const size_t nElements)
{
    // Set the alpha field to max for all pixels
    // Note: little endian representation (A,R,G,B)
    for (size_t idx = 0; idx < nElements; idx++)
        data[idx] = 0xFF000000; 
}

int main()
{
    const MatrixDim DIMENSIONS = {768, 768, 2};
    const size_t N_ELEMENTS = DIMENSIONS.x * DIMENSIONS.y;
    const size_t FIELD_SIZE = sizeof(float) * N_ELEMENTS * DIMENSIONS.vl;
    const size_t BGR_SIZE = sizeof(unsigned int) * N_ELEMENTS;
    const float RDX = 512.0;

    // Simulation timestep
    const float TIMESTEP = 0.01;

    // Rendering frame rate (milliseconds)
    const int FRAMERATE = 1;

    // Setup CUDA Grids and Blocks
    const dim3 DIM_BLOCK(32,32); // This is the maximum as per CUDA 2.x
    const dim3 DIM_GRID(
        (DIMENSIONS.x + DIM_BLOCK.x - 1) / DIM_BLOCK.x,
        (DIMENSIONS.y + DIM_BLOCK.y - 1) / DIM_BLOCK.y);

    // Setup host velocity field
    float *h_vfield;
    cudaMallocHost(&h_vfield, FIELD_SIZE);
    initialize_p_field(h_vfield, DIMENSIONS);

    // Setup device velocity, pressure and divergence field
    float *d_vfield;
    float *d_pfield;
    float *d_dfield;
    cudaMalloc(&d_vfield, FIELD_SIZE);
    cudaMalloc(&d_pfield, FIELD_SIZE);
    cudaMalloc(&d_dfield, FIELD_SIZE);

    // Setup host image matrix
    unsigned int *h_bgr;
    cudaMallocHost(&h_bgr, BGR_SIZE);
    initialize_bgr_field(h_bgr, N_ELEMENTS);

    // Setup device image matrix
    unsigned int *d_bgr;
    cudaMalloc(&d_bgr, BGR_SIZE);
    cudaMemcpy(d_bgr, h_bgr, BGR_SIZE, cudaMemcpyHostToDevice);

    cv::Mat image;
    size_t temp = 0;
    while (true)
    {
        cudaMemcpy(d_vfield, h_vfield, FIELD_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vfield, d_vfield, FIELD_SIZE, cudaMemcpyDeviceToDevice);
        
        kernel_advect<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_vfield,d_vfield,RDX,TIMESTEP,0.8);
        kernel_divergence<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_vfield,d_dfield,RDX/2.0);

        //kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(d_vfield, d_bgr, DIMENSIONS);
        kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(d_dfield, d_bgr, DIMENSIONS);
        cudaMemcpy(h_vfield, d_vfield, FIELD_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bgr, d_bgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        image = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_bgr);

        cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display Image", image);
        cv::waitKey(FRAMERATE);

        if (temp > 500)
        {
            temp = 0;
            initialize_p_field(h_vfield, DIMENSIONS);
        }
        else
        {
            temp++;
        }
    }

    cudaFree(h_vfield);
    cudaFree(h_bgr);
    cudaFree(d_vfield);
    cudaFree(d_pfield);
    cudaFree(d_dfield);
    cudaFree(d_bgr);
    return 0;
}