#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "include/gradient.cuh"
#include "include/fluid_sim.cuh"

static float rand_norm_scalar()
{
    float retval = (float) rand() / (float) RAND_MAX + 1.0;
    retval = retval * ((rand() % 2 > 0) ? 1.0 : -1.0);
    return retval;
}

static __host__ void initialize_p_field(float *data, const size_t nRows, const size_t nCols)
{
    const size_t radius = rand() % 100;
    const size_t x_lower = nCols/2 - radius;
    const size_t x_upper = nCols/2 + radius;
    const size_t y_lower = nRows/2 - radius;
    const size_t y_upper = nRows/2 + radius;

    for (size_t y = y_lower; y < y_upper; y++)
    {
        for (size_t x = x_lower; x < x_upper; x++)
        {
            data[(y * nCols + x) * 2] = rand_norm_scalar();
            data[(y * nCols + x) * 2 + 1] = rand_norm_scalar();
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
    const size_t HEIGHT = 512;
    const size_t WIDTH = 512;
    const size_t DIMENSIONS = 2;
    const size_t N_ELEMENTS = HEIGHT * WIDTH;
    const size_t FIELD_SIZE = sizeof(float) * N_ELEMENTS * DIMENSIONS;
    const size_t BGR_SIZE = sizeof(unsigned int) * N_ELEMENTS;
    const size_t RDX = HEIGHT / 2;

    // Simulation timestep
    const float TIMESTEP = 0.001;

    // Rendering frame rate (milliseconds)
    const int FRAMERATE = 10;

    // Setup CUDA Grids and Blocks
    const dim3 DIM_BLOCK(32,32); // This is the maximum as per CUDA 2.x
    const dim3 DIM_GRID(
        (WIDTH + DIM_BLOCK.x - 1) / DIM_BLOCK.x,
        (HEIGHT + DIM_BLOCK.y - 1) / DIM_BLOCK.y);

    // Setup host pressure field
    float *h_pfield;
    cudaMallocHost(&h_pfield, FIELD_SIZE);
    initialize_p_field(h_pfield, HEIGHT, WIDTH);

    // Setup device pressure field
    float *d_pfield;
    float *d_pfield_temp;
    cudaMalloc(&d_pfield, FIELD_SIZE);
    cudaMalloc(&d_pfield_temp, FIELD_SIZE);

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
        cudaMemcpy(d_pfield, h_pfield, FIELD_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pfield_temp, d_pfield, FIELD_SIZE, cudaMemcpyDeviceToDevice);
        kernel_advect<<<DIM_GRID, DIM_BLOCK>>>(WIDTH, HEIGHT, d_pfield, d_pfield_temp, RDX, TIMESTEP);
        cudaMemcpy(d_pfield, d_pfield_temp, FIELD_SIZE, cudaMemcpyDeviceToDevice);
        kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(d_pfield, d_bgr, WIDTH, HEIGHT);
        cudaMemcpy(h_pfield, d_pfield, FIELD_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bgr, d_bgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        image = cv::Mat(HEIGHT, WIDTH, CV_8UC4, (unsigned *)h_bgr);
#if 0
        for (size_t idx = 0; idx < N_ELEMENTS; idx++)
        {
            printf("%x\n",h_bgr[idx]);
        }
#endif
#if 0
        printf("iteration: %d\n",temp);
        for (size_t idx = 0; idx < N_ELEMENTS; idx++)
        {
            printf("(%f %f) ", h_pfield[idx * 2], h_pfield[idx * 2+1]);
            if (idx % WIDTH == 0)
                printf("\n");
        }
        for (size_t idx = 0; idx < N_ELEMENTS; idx++)
        {
            printf("0x%x ", h_bgr[idx]);
            if (idx % WIDTH == 0)
                printf("\n");
        }
#endif
        // printf("0x%X\n",normalized2bgr(1.0));
        //printf("0x%X\n", data[1]);
        //printf("%f\n",rand_norm_scalar());

        cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display Image", image);
        cv::waitKey(FRAMERATE);

        if (temp > 50)
        {
            temp = 0;
            initialize_p_field(h_pfield, HEIGHT, WIDTH);
        }
        else
        {
            temp++;
        }
    }

    cudaFree(h_pfield);
    cudaFree(d_pfield);
    cudaFree(h_bgr);
    return 0;
}