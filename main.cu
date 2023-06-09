//-----------------------------------------------------------------------------
/// @file main.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief CUDA Fluid Simulation Main Driver
//-----------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "include/window_utils.cuh"
#include "include/fluid_sim.cuh"
#include "include/fluid_utils.cuh"

static size_t pulse_x;
static size_t pulse_y;
static bool pulse = false;

//-----------------------------------------------------------------------------
/// @brief OpenCV Mouse Handler function
/// @param event handler event enumeration
/// @param x x window coordinate
/// @param y y window coordinate
/// @param flags mouse flags (unused)
/// @param params user parameters (unused)
//-----------------------------------------------------------------------------
static void mouse_pulse(int event, int x, int y, int flags, void *params)
{
    pulse_x = x;
    pulse_y = y;
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

//-----------------------------------------------------------------------------
/// @brief Performs Fluid Simulation Timestep for a single frame.
/// @param cudim_grid dim3 definition of the CUDA grid dimensions
/// @param cudim_block dim3 definition of the CUDA block dimensions (32x32)
/// @param dim working vector space dimensions
/// @param d_vfield velocity vector field matrix
/// @param d_dfield divergence scalar field matrix
/// @param d_pfield pressure scalar field matrix
/// @param rdx space factor constant
/// @param timestep timestep factor
/// @param jacobi_iterations number of iterations to perform for jacobi solutions
//-----------------------------------------------------------------------------
static void fluid_sim_frame(
    const dim3 cudim_grid, const dim3 cudim_block, const MatrixDim dim,
    float *d_vfield, float *d_dfield, float *d_pfield,
    const float rdx, const float timestep, const size_t jacobi_iterations)
{
    // ----- COMPUTE ADVECTION ----- //
    kernel_advect<<<cudim_grid, cudim_block>>>(dim, d_vfield, d_vfield, rdx, timestep, 0.9);

    // ----- COMPUTE DIVERGENCE ----- //
    kernel_divergence<<<cudim_grid, cudim_block>>>(dim, d_vfield, d_dfield, 0.5);

    // ----- COMPUTE PRESSURE ----- //
    cudaMemset(d_pfield, 0, sizeof(float) * dim.x * dim.y);
    for (size_t j_iter = 0; j_iter < jacobi_iterations; j_iter++)
    {
        kernel_sboundary<<<cudim_grid, cudim_block>>>(dim, d_pfield, 1.0);
        kernel_jacobi<<<cudim_grid, cudim_block>>>(dim, d_pfield, d_dfield, -1.0, 0.25);
    }

    // ----- COMPUTE BOUNDARIES ----- //
    kernel_vboundary<<<cudim_grid, cudim_block>>>(dim, d_vfield, -1.0);

    // ----- COMPUTE PRESSURE-VELOCITY GRADIENT ----- //
    /// @todo this step is unstable at higher values of rdx
    kernel_gradient<<<cudim_grid, cudim_block>>>(dim, d_pfield, d_vfield, 0.5);
}

//-----------------------------------------------------------------------------
/// @brief Main Function Driver
/// @return 0
//-----------------------------------------------------------------------------
int main()
{
    // ----- CONSTANTS ----- //

    /// @brief Defines the working and rendering dimensions for the simulator
    const MatrixDim DIMENSIONS = {576, 576, 2};
    
    /// @brief Defines the number of jacobi iterations to perform for the
    /// Pressure Gradient Computation step.
    const size_t JACOBI_ITERATIONS = 20;

    /// @brief Defines the scalar quanitity for the Advection Computation.
    /// Default: 256.0
    const float RDX = 256.0;

    // Simulation timestep
    const float TIMESTEP = 0.05;

    // Rendering frame rate (milliseconds)
    const int FRAMERATE = 1;

    // Setup CUDA Grids and Blocks
    const dim3 DIM_BLOCK(32, 32); // This is the maximum as per CUDA 2.x
    const dim3 DIM_GRID(
        (DIMENSIONS.x + DIM_BLOCK.x - 1) / DIM_BLOCK.x,
        (DIMENSIONS.y + DIM_BLOCK.y - 1) / DIM_BLOCK.y);

    // ----- FLUID SIMULATION ----- //

    /// @brief Device Buffer Pointer for the Fluid Velocity Field.
    /// The Fluid Velocity Field is a Matrix of 2-Dimensional Vectors.
    const size_t VECTOR_FIELD_SIZE = sizeof(float) * DIMENSIONS.x * DIMENSIONS.y * DIMENSIONS.vl;
    float *d_vfield, *h_vfield;
    cudaMalloc(&d_vfield, VECTOR_FIELD_SIZE); cudaMallocHost(&h_vfield, VECTOR_FIELD_SIZE);

    /// @brief Device Buffer Pointer for the Fluid Divergence Field.
    /// The Fluid Divergence Field a a Matrix of Scalar Values.
    const size_t SCALAR_FIELD_SIZE = sizeof(float) * DIMENSIONS.x * DIMENSIONS.y;
    float *d_dfield, *h_dfield;
    cudaMalloc(&d_dfield, SCALAR_FIELD_SIZE); cudaMallocHost(&h_dfield, SCALAR_FIELD_SIZE);

    /// @brief Device Buffer Pointer for the Fluid Divergence Field.
    /// The Fluid Divergence Field a a Matrix of Scalar Values.
    float *d_pfield, *h_pfield;
    cudaMalloc(&d_pfield, SCALAR_FIELD_SIZE); cudaMallocHost(&h_pfield, SCALAR_FIELD_SIZE);
    
    // ----- BGR WINDOW RENDERING ----- //
    // Setup host and device GBR matrix buffers
    unsigned int *h_vbgr, *d_vbgr;
    unsigned int *h_dbgr, *d_dbgr;
    unsigned int *h_pbgr, *d_pbgr;
    const size_t BGR_SIZE = sizeof(unsigned int) * DIMENSIONS.x * DIMENSIONS.y;
    cudaMallocHost(&h_vbgr, BGR_SIZE); cudaMalloc(&d_vbgr, BGR_SIZE);
    cudaMallocHost(&h_dbgr, BGR_SIZE); cudaMalloc(&d_dbgr, BGR_SIZE);
    cudaMallocHost(&h_pbgr, BGR_SIZE); cudaMalloc(&d_pbgr, BGR_SIZE);    

    // Setup OpenCV Windows
    cv::Mat vimage; cv::namedWindow("Velocity", cv::WINDOW_AUTOSIZE);
    cv::Mat dimage; cv::namedWindow("Divergence", cv::WINDOW_AUTOSIZE);
    cv::Mat pimage; cv::namedWindow("Pressure", cv::WINDOW_AUTOSIZE);
    
    // ----- WINDOW MOUSE HANDLER HOOK ----- //
    cv::setMouseCallback("Velocity", mouse_pulse, 0);

    printf("CUDA FLUID SIMULATOR\n");
    printf("AUTHOR: NATE LAO (NLAO1@JH.EDU)\n");
    printf("--------------------\n");
    // ----- FRAME LOOP ----- //
    while (true)
    {
        const std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();

        // ----- PERFORM FLUID SIMULATION ----- //
        fluid_sim_frame(
            DIM_GRID, DIM_BLOCK, DIMENSIONS,
            d_vfield, d_dfield, d_pfield,
            RDX, TIMESTEP, JACOBI_ITERATIONS);

        // ----- COPY TO HOST ----- //
        cudaMemcpy(h_vfield, d_vfield, VECTOR_FIELD_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dfield, d_dfield, SCALAR_FIELD_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pfield, d_pfield, SCALAR_FIELD_SIZE, cudaMemcpyDeviceToHost);

        // ----- CONVERT TO BGR ----- //
        kernel_vfield2bgr<<<DIM_GRID, DIM_BLOCK>>>(d_vfield, d_vbgr, DIMENSIONS); // Advection
        kernel_sfield2bgr<<<DIM_GRID, DIM_BLOCK>>>(d_dfield, d_dbgr, DIMENSIONS); // Divergence
        kernel_sfield2bgr<<<DIM_GRID, DIM_BLOCK>>>(d_pfield, d_pbgr, DIMENSIONS); // Pressure
        cudaMemcpy(h_vbgr, d_vbgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dbgr, d_dbgr, BGR_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pbgr, d_pbgr, BGR_SIZE, cudaMemcpyDeviceToHost);

        // ----- RENDER TO IMAGE ----- //
        vimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_vbgr);
        dimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_dbgr);
        pimage = cv::Mat(DIMENSIONS.y, DIMENSIONS.x, CV_8UC4, (unsigned *)h_pbgr);
        cv::imshow("Velocity", vimage);
        cv::imshow("Divergence", dimage);
        cv::imshow("Pressure", pimage);
        cv::waitKey(FRAMERATE);

        const std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
        printf("FPS: %.02f Velocity: (%+.05f, %+.05f) Divergence: %+.05f Pressure: %+0.5f\r",
            (1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()),
            h_vfield[matrix_index(pulse_x, pulse_y, DIMENSIONS, 0)],
            h_vfield[matrix_index(pulse_x, pulse_y, DIMENSIONS, 1)],
            h_dfield[pulse_y * DIMENSIONS.x + pulse_x],
            h_pfield[pulse_y * DIMENSIONS.x + pulse_x]);

        // ----- PULSE VELOCITY ON WINDOW CLICK ----- //
        if (pulse)
        {
            kernel_pulse<<<DIM_GRID, DIM_BLOCK>>>(pulse_x, pulse_y, d_vfield, DIMENSIONS, 0.5);
        }
    }

    cudaFree(d_vfield);
    cudaFree(d_dfield);
    cudaFree(d_pfield);
    cudaFree(h_vfield);
    cudaFree(h_dfield);
    cudaFree(h_pfield);
    cudaFree(d_vbgr);
    cudaFree(d_dbgr);
    cudaFree(d_pbgr);
    cudaFree(h_vbgr);
    cudaFree(h_pbgr);
    cudaFree(h_dbgr);
    return 0;
}