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

__host__ __device__ float divergence(
    const size_t x, const size_t y,
    const float *data,
    const MatrixDim dim,
    const float halfrdx);

__global__ void kernel_jacobi(
    const MatrixDim dim,
    float *X,
    const float *B,
    const float alpha,
    const float beta);

__host__ __device__ float jacobi(
    const size_t x, const size_t y,
    const float *X,
    const float *B,
    const MatrixDim dim,
    const float alpha,
    const float beta);

__global__ void kernel_sboundary(
    const MatrixDim dim,
    float *M,
    const float scale);

__global__ void kernel_vboundary(
    const MatrixDim dim,
    float *M,
    const float scale);

__global__ void kernel_gradient(
    const MatrixDim dim,
    const float *P,
    float *V,
    const float halfrdx);
#endif