#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "include/gradient.cuh"
#include "include/fluid_sim.cuh"
#include "include/fluid_utils.cuh"

#if 0
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
#if 1
            data[matrix_index(x,y,dim,0)] = 2.0 * rand_norm_scalar();
            data[matrix_index(x,y,dim,1)] = 2.0;// * rand_norm_scalar();
#else
            data[matrix_index(x,y,dim,0)] = rand_norm_scalar();
            data[matrix_index(x,y,dim,1)] = rand_norm_scalar();
#endif
        }
    }
}
#endif

struct Coordinate
{
    size_t x;
    size_t y;
};
static Coordinate pulse_coordinate;
static bool pulse = false;

void mouse_pulse(int event, int x, int y, int flags, void *coordinate)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        //printf("EVENT_LBUTTONDOWN\n");
        pulse_coordinate.x = x;
        pulse_coordinate.y = y;
        pulse = true;
        break;
    }
}

int main()
{
    const MatrixDim DIMENSIONS = {768, 768, 2};
    const size_t N_ELEMENTS = DIMENSIONS.x * DIMENSIONS.y;
    const size_t FIELD_SIZE = sizeof(float) * N_ELEMENTS * DIMENSIONS.vl;
    const size_t BGR_SIZE = sizeof(unsigned int) * N_ELEMENTS;
    const float RDX = 256.0;

    // Simulation timestep
    const float TIMESTEP = 0.05;

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
    //initialize_p_field(h_vfield, DIMENSIONS);

    // Setup device velocity, pressure and divergence field
    float *d_vfield;
    float *d_pfield;
    float *d_dfield;
    cudaMalloc(&d_vfield, FIELD_SIZE);
    cudaMalloc(&d_pfield, FIELD_SIZE);
    cudaMalloc(&d_dfield, FIELD_SIZE);

    // Setup host image matrix
    unsigned int *h_vbgr;
    unsigned int *h_pbgr;
    unsigned int *h_dbgr;
    cudaMallocHost(&h_vbgr, BGR_SIZE);
    cudaMallocHost(&h_pbgr, BGR_SIZE);
    cudaMallocHost(&h_dbgr, BGR_SIZE);

    // Setup device image matrix
    unsigned int *d_vbgr;
    unsigned int *d_pbgr;
    unsigned int *d_dbgr;
    cudaMalloc(&d_vbgr, BGR_SIZE);
    cudaMalloc(&d_pbgr, BGR_SIZE);
    cudaMalloc(&d_dbgr, BGR_SIZE);

    cv::Mat vimage;
    cv::Mat pimage;
    cv::Mat dimage;
    cv::namedWindow("Velocity", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Pressure", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Divergence", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Velocity", mouse_pulse, 0);
    size_t temp = 0;
    while (true)
    {
        //cudaMemcpy(d_vfield, h_vfield, FIELD_SIZE, cudaMemcpyHostToDevice);
        cudaMemset(d_pfield,0, FIELD_SIZE);

        kernel_advect<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_vfield,d_vfield,RDX,TIMESTEP,0.9);
        kernel_divergence<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_vfield,d_dfield,RDX/2.0);
        kernel_jacobi<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_pfield,d_dfield,d_pfield,-1.0,0.25,20);

        kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(d_vfield, d_vbgr, DIMENSIONS);
        kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(d_pfield, d_pbgr, DIMENSIONS);
        kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(d_dfield, d_dbgr, DIMENSIONS);
        cudaMemcpy(h_vbgr, d_vbgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pbgr, d_pbgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dbgr, d_dbgr, BGR_SIZE, cudaMemcpyDeviceToHost);

        //cudaMemcpy(h_vfield, d_vfield, FIELD_SIZE, cudaMemcpyDeviceToHost);
        //cudaDeviceSynchronize();

        vimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_vbgr);
        pimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_pbgr);
        dimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_dbgr);
        cv::imshow("Velocity", vimage);
        cv::imshow("Pressure", pimage);
        cv::imshow("Divergence", dimage);
        cv::waitKey(FRAMERATE);

        if (pulse)
        {
            kernel_pulse<<<DIM_GRID, DIM_BLOCK>>>(pulse_coordinate.x, pulse_coordinate.y,d_vfield,DIMENSIONS);
            pulse = false;
        }

#if 0
        if (temp > 100)
        {
            temp = 0;
            //initialize_p_field(h_vfield, DIMENSIONS);
            kernel_pulse<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS.x/2, DIMENSIONS.y /2,d_vfield,DIMENSIONS);
        }
        else
        {
            temp++;
        }
#endif
    }

    cudaFree(h_vfield);
    cudaFree(h_vbgr);
    cudaFree(h_pbgr);
    cudaFree(h_dbgr);
    cudaFree(d_vfield);
    cudaFree(d_pfield);
    cudaFree(d_dfield);
    cudaFree(d_vbgr);
    cudaFree(d_pbgr);
    cudaFree(d_dbgr);
    return 0;
}