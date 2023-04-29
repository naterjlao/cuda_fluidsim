#ifndef __GRADIENT_CUH__
#define __GRADIENT_CUH__
#include "fluid_utils.cuh"
__global__ void kernel_gradient(
    const float *field, 
    unsigned int *bgr,
    const MatrixDim dim);

__global__ void kernel_pulse(
    const size_t epicenter_x, const size_t epicenter_y,
    float *field, const MatrixDim dim);

#endif