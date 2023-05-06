#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

#include "include/window_utils.cuh"
#include "include/fluid_sim.cuh"
#include "include/fluid_utils.cuh"

struct Coordinate
{
    size_t x;
    size_t y;
};
static Coordinate pulse_coordinate;
static bool pulse = false;

void mouse_pulse(int event, int x, int y, int flags, void *coordinate)
{
    pulse_coordinate.x = x;
    pulse_coordinate.y = y;
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        pulse = true;
        break;
    case cv::EVENT_LBUTTONUP:
        pulse = false;
        break;
    }
}

int main()
{
    //const MatrixDim DIMENSIONS = {1024, 1024, 2};
    const MatrixDim DIMENSIONS = {512, 512, 2};
    const size_t SCALAR_FIELD_SIZE = sizeof(float) * DIMENSIONS.x * DIMENSIONS.y;
    const size_t VECTOR_FIELD_SIZE = sizeof(float) * DIMENSIONS.x * DIMENSIONS.y * DIMENSIONS.vl;
    const size_t BGR_SIZE = sizeof(unsigned int) * DIMENSIONS.x * DIMENSIONS.y;
    const size_t JACOBI_ITERATIONS = 20;
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

    // Setup device velocity, pressure and divergence fields
    float *d_vfield;
    float *d_dfield;
    float *d_pfield;
    cudaMalloc(&d_vfield, VECTOR_FIELD_SIZE);
    cudaMalloc(&d_dfield, SCALAR_FIELD_SIZE);
    cudaMalloc(&d_pfield, SCALAR_FIELD_SIZE);

    // Setup host image matrix
    unsigned int *h_vbgr;
    unsigned int *h_dbgr;
    unsigned int *h_pbgr;
    cudaMallocHost(&h_vbgr, BGR_SIZE);
    cudaMallocHost(&h_dbgr, BGR_SIZE);
    cudaMallocHost(&h_pbgr, BGR_SIZE);

    // Setup device image matrix
    unsigned int *d_vbgr;
    unsigned int *d_dbgr;
    unsigned int *d_pbgr;
    cudaMalloc(&d_vbgr, BGR_SIZE);
    cudaMalloc(&d_dbgr, BGR_SIZE);
    cudaMalloc(&d_pbgr, BGR_SIZE);

    cv::Mat vimage;
    cv::Mat dimage;
    cv::Mat pimage;
    cv::namedWindow("Velocity", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Divergence", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Pressure", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Velocity", mouse_pulse, 0);
    while (true)
    {
        // ----- COMPUTE ADVECTION ----- //
        kernel_advect<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_vfield,d_vfield,RDX,TIMESTEP,0.9);

        // ----- COMPUTE DIVERGENCE ----- //
        kernel_divergence<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_vfield,d_dfield,RDX/2.0);

        // ----- COMPUTE PRESSURE ----- //
        cudaMemset(d_pfield,0, SCALAR_FIELD_SIZE);
        for (size_t j_iter = 0; j_iter < JACOBI_ITERATIONS; j_iter++)
        {
            kernel_sboundary<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS, d_pfield,1.0);
            kernel_jacobi<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS,d_pfield,d_dfield,-1.0,0.01);
        }

        // ----- COMPUTE BOUNDARIES ----- //
        kernel_vboundary<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS, d_vfield, -1.0);

        // ----- COMPUTE PRESSURE-VELOCITY GRADIENT ----- //
        kernel_gradient<<<DIM_GRID, DIM_BLOCK>>>(DIMENSIONS, d_pfield, d_vfield, 0.001);

        // ----- CONVERT TO BGR ----- //
        kernel_vfield2bgr<<<DIM_GRID, DIM_BLOCK>>>(d_vfield, d_vbgr, DIMENSIONS); // Advection
        kernel_sfield2bgr<<<DIM_GRID, DIM_BLOCK>>>(d_dfield, d_dbgr, DIMENSIONS); // Divergence
        kernel_sfield2bgr<<<DIM_GRID, DIM_BLOCK>>>(d_pfield, d_pbgr, DIMENSIONS); // Pressure
        cudaMemcpy(h_vbgr, d_vbgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dbgr, d_dbgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pbgr, d_pbgr, BGR_SIZE, cudaMemcpyDeviceToHost);

        vimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_vbgr);
        dimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_dbgr);
        pimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_pbgr);
        cv::imshow("Velocity", vimage);
        cv::imshow("Divergence", dimage);
        cv::imshow("Pressure", pimage);
        cv::waitKey(FRAMERATE);

        if (pulse)
        {
            kernel_pulse<<<DIM_GRID, DIM_BLOCK>>>(pulse_coordinate.x, pulse_coordinate.y,d_vfield,DIMENSIONS,0.5);
        }
    }

    cudaFree(d_vfield);
    cudaFree(d_dfield);
    cudaFree(d_pfield);
    cudaFree(d_vbgr);
    cudaFree(d_dbgr);
    cudaFree(d_pbgr);
    cudaFree(h_vbgr);
    cudaFree(h_pbgr);
    cudaFree(h_dbgr);
    return 0;
}