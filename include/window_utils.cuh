//-----------------------------------------------------------------------------
/// @file windows_utils.cuh
/// @author Nate Lao (nlao1@jh.edu)
/// @brief CUDA Fluid Simulation Window Utilities
//-----------------------------------------------------------------------------
#ifndef __GRADIENT_CUH__
#define __GRADIENT_CUH__
#include "fluid_utils.cuh"

__global__ void kernel_vfield2bgr(
    const float *field, 
    unsigned int *bgr,
    const MatrixDim dim);

__global__ void kernel_sfield2bgr(
    const float *field,
    unsigned int *bgr,
    const MatrixDim dim);

__global__ void kernel_pulse(
    const size_t epicenter_x, const size_t epicenter_y,
    float *field, const MatrixDim dim, const float intensity);

#endif