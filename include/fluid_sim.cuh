#ifndef __FLUID_SIM_CUH__
#define __FLUID_SIM_CUH__

#include "fluid_utils.cuh"

__global__ void kernel_advect(
    const MatrixDim dim,
    const float* input_data,
    float* output_data,
    const size_t rdx,
    const float timestep,
    const float dissipation = 0.999);

__device__ void advect(
    const MatrixDim dim,
    const size_t x,
    const size_t y,
    const size_t rdx,
    const float timestep,
    const float dissipation,
    const float *u_matrix,
    const float *d_matrix,
    float *dx_new,
    float *dy_new);
#endif