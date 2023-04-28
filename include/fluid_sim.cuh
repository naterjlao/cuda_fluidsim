#ifndef __FLUID_SIM_CUH__
#define __FLUID_SIM_CUH__

#include "fluid_utils.cuh"

__global__ void kernel_advect(
    const MatrixDim dim,
    const float* input_data,
    float* output_data,
    const float rdx,
    const float timestep,
    const float dissipation = 0.999);

__host__ __device__ void advect(
    const MatrixDim dim,
    const size_t x,
    const size_t y,
    const float rdx,
    const float timestep,
    const float dissipation,
    const float *u_matrix,
    const float *d_matrix,
    float *dx_new,
    float *dy_new);

__global__ void kernel_divergence(
    const MatrixDim dim,
    const float *velocity,
    float *div,
    const float halfrdx);

__host__ __device__ Vector divergence(
    const size_t x, const size_t y,
    const float *data,
    const MatrixDim dim,
    const float halfrdx);

__global__ void kernel_jacobi(
    const MatrixDim dim,
    const float *X,
    const float *B,
    float *X_new,
    const float alpha,
    const float beta,
    const size_t iterations=1);

__host__ __device__ Vector jacobi(
    const size_t x, const size_t y,
    const float *x_vector,
    const float *b_vector,
    const MatrixDim dim,
    const float alpha,
    const float beta);
#endif