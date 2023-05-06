//-----------------------------------------------------------------------------
/// @file fluid_utils.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief CUDA Fluid Simulation Utilities
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include "include/fluid_utils.cuh"

//-----------------------------------------------------------------------------
/// @brief Computes the Matrix Index for a Multidimensional Vector Matrix
/// @param[in] x x index coordinate
/// @param[in] y y index coordinate
/// @param[in] dim Defines the working space dimension bounds.
/// @param[in] vc vector component, 0->x/1->y
/// @return matrix index of the desired vector value.
//-----------------------------------------------------------------------------
__host__ __device__ size_t matrix_index(const size_t x, const size_t y, const MatrixDim dim, const size_t vc)
{
    return (y * dim.x + x) * dim.vl + vc;
}

//-----------------------------------------------------------------------------
/// @brief Performs a bilinear interpolation given (px, py) and vector data field.
/// @details
/// (idx_lo, idy_lo) --- * ------ (idx_hi, idy_lo)
///                      |
///                      * <- [vx, vy]
///                      |
///                      |
///                      |
/// (idx_lo, idy_hi) --- * ------ (idx_hi, idy_hi)
/// @param[in] px x interpolated coordinate
/// @param[in] py y interpolated coordinate
/// @param[in] data data matrix (see note)
/// @param[in] dim Defines the working space dimension bounds.
/// @param[out] vx interpolation result of x vector component
/// @param[out] vy interpolation result of the y vector component.
/// @return None.
/// @note the data matrix must be defined as a multi-dimensions
/// array of [dim.y][dim.x][2] (2 is the vector length)
//-----------------------------------------------------------------------------
__host__ __device__ void bilinear_interpolation(
    const float px, const float py,
    const float *data, const MatrixDim dim,
    float *vx, float *vy)
{
    const size_t idx_hi = (size_t)ceil(px);
    const size_t idx_lo = (size_t)floor(px);
    const size_t idy_hi = (size_t)ceil(py);
    const size_t idy_lo = (size_t)floor(py);

    if ((idx_hi < dim.x) && (idy_hi < dim.y))
    {
        const float factor_h = px - floor(px); // Horizontal Ratio Factor
        const float factor_v = py - floor(py); // Vertical Ratio Factor

        // North Interpolation
        const float north_vx =
            (data[matrix_index(idx_lo, idy_lo, dim, 0)] * (1.0 - factor_h)) + // West
            (data[matrix_index(idx_hi, idy_lo, dim, 0)] * (factor_h));        // East
        const float north_vy =
            (data[matrix_index(idx_lo, idy_lo, dim, 1)] * (1.0 - factor_h)) + // West
            (data[matrix_index(idx_hi, idy_lo, dim, 1)] * (factor_h));        // East

        // South Interpolation
        const float south_vx =
            (data[matrix_index(idx_lo, idy_hi, dim, 0)] * (1.0 - factor_h)) + // West
            (data[matrix_index(idx_hi, idy_hi, dim, 0)] * (factor_h));        // East
        const float south_vy =
            (data[matrix_index(idx_lo, idy_hi, dim, 1)] * (1.0 - factor_h)) + // West
            (data[matrix_index(idx_hi, idy_hi, dim, 1)] * (factor_h));        // East

        // Center Interpolation
        *vx = north_vx * (1.0 - factor_v) + south_vx * (factor_v);
        *vy = north_vy * (1.0 - factor_v) + south_vy * (factor_v);
    }
}

//-----------------------------------------------------------------------------
/// @brief Retrieves the bounding neighbor vectors given a coordinate position.
/// @param[in] x x coordinate
/// @param[in] y y coordinate
/// @param[in] data data matrix (see note)
/// @param[in] dim Defines the working space dimension bounds.
/// @param[out] vN North Vector
/// @param[out] vS South Vector
/// @param[out] vE East Vector
/// @param[out] vW West Vector
/// @return None.
/// @note the data matrix must be defined as a multi-dimensions
/// array of [dim.y][dim.x][2] (2 is the vector length)
//-----------------------------------------------------------------------------
__host__ __device__ void neighbors_vector(
    const size_t x, const size_t y,
    const float *data, const MatrixDim dim,
    Vector *vN, Vector *vS, Vector *vE, Vector *vW)
{
    if ((x < dim.x) && (y < dim.y))
    {
        const Vector vC = {
            .x = data[matrix_index(x, y, dim, 0)],
            .y = data[matrix_index(x, y, dim, 1)]};

        // North Neighbor Vector
        vN->x = (y > 0) ? data[matrix_index(x, y - 1, dim, 0)] : vC.x;
        vN->y = (y > 0) ? data[matrix_index(x, y - 1, dim, 1)] : vC.y;

        // South Neighbor Vector
        vS->x = (y < (dim.y - 1)) ? data[matrix_index(x, y + 1, dim, 0)] : vC.x;
        vS->y = (y < (dim.y - 1)) ? data[matrix_index(x, y + 1, dim, 1)] : vC.y;

        // East Neighbor Vector
        vE->x = (x < (dim.x - 1)) ? data[matrix_index(x + 1, y, dim, 0)] : vC.x;
        vE->y = (x < (dim.x - 1)) ? data[matrix_index(x + 1, y, dim, 1)] : vC.y;

        // West Neighbor Vector
        vW->x = (x > 0) ? data[matrix_index(x - 1, y, dim, 0)] : vC.x;
        vW->y = (x > 0) ? data[matrix_index(x - 1, y, dim, 1)] : vC.y;
    }
}

//-----------------------------------------------------------------------------
/// @brief Retrieves the bounding neighbor scalar values given a coordinate position.
/// @param[in] x x coordinate
/// @param[in] y y coordinate
/// @param[in] data data matrix (see note)
/// @param[in] dim Defines the working space dimension bounds.
/// @param[out] sN North Component
/// @param[out] sS South Component
/// @param[out] sE East Component
/// @param[out] sW West Component
/// @return None.
/// @note the data matrix must be defined as a scalar
/// array of [dim.y][dim.x]
//-----------------------------------------------------------------------------
__host__ __device__ void neighbors_scalar(
    const size_t x, const size_t y,
    const float *data, const MatrixDim dim,
    float *sN, float *sS, float *sE, float *sW)
{
    if ((x < dim.x) && (y < dim.y))
    {
        const float sC = data[y * dim.x + x];

        // North Neighbor Vector
        *sN = (y > 0) ? data[(y - 1) * dim.x + x] : sC;

        // South Neighbor Vector
        *sS = (y < (dim.y - 1)) ? data[(y + 1) * dim.x + x] : sC;

        // East Neighbor Vector
        *sE = (x < (dim.x - 1)) ? data[y * dim.x + (x + 1)] : sC;

        // West Neighbor Vector
        *sW = (x > 0) ? data[y * dim.x + (x - 1)] : sC;
    }
}
