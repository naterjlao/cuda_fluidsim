//-----------------------------------------------------------------------------
/// @file fluid_utils.cuh
/// @author Nate Lao (nlao1@jh.edu)
/// @brief CUDA Fluid Simulation Utilities
//-----------------------------------------------------------------------------
#ifndef __FLUID_UTILS_CUH__
#define __FLUID_UTILS_CUH__

struct MatrixDim
{
    size_t x;
    size_t y;
    size_t vl;
};

struct Vector
{
    float x;
    float y;

    __host__ __device__ Vector operator+(Vector rh) const
    {
        return {x + rh.x, y + rh.y};
    }

    __host__ __device__ Vector operator*(const float scalar) const
    {
        return {x * scalar, y * scalar};
    }
};

__host__ __device__ size_t matrix_index(const size_t x, const size_t y, const MatrixDim dim, const size_t vc);

__host__ __device__ void bilinear_interpolation(
    const float px, const float py,
    const float *data, const MatrixDim dim,
    float *vx, float *vy);

__host__ __device__ void neighbors_vector(
    const size_t x, const size_t y,
    const float *data, const MatrixDim dim,
    Vector *vN, Vector *vS, Vector *vE, Vector *vW);

__host__ __device__ void neighbors_scalar(
    const size_t x, const size_t y,
    const float *data, const MatrixDim dim,
    float *sN, float *sS, float *sE, float *sW);

#endif