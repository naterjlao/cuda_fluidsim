#ifndef __FLUID_UTILS_CUH__
#define __FLUID_UTILS_CUH__

typedef struct
{
    size_t x;
    size_t y;
    size_t vl;
} MatrixDim;

__host__ __device__ bool bilinear_interpolation(
    const float px, const float py,
    const float *data, const MatrixDim dim,
    float *vx, float *vy);
#endif