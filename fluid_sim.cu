#include "include/fluid_sim.cuh"

__global__ void kernel_advect(
    const MatrixDim dim,
    const float *input_data,
    float *output_data,
    const size_t rdx,
    const float timestep,
    const float dissipation)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < dim.x) && (y < dim.y))
  {
    float dx_new = output_data[matrix_index(x, y, dim, 0)];
    float dy_new = output_data[matrix_index(x, y, dim, 1)];

    advect(dim, x, y, rdx, timestep, dissipation,
           input_data, output_data, &dx_new, &dy_new);

    output_data[matrix_index(x, y, dim, 0)] = dx_new;
    output_data[matrix_index(x, y, dim, 1)] = dy_new;
  }
}

/// @brief
/// @param dim_x Dimension x size of the input matrices
/// @param dim_y Dimension y size of the input matrices
/// @param coord_x Position x coordinate
/// @param coord_y Position y coordinate
/// @param timestep Timestep
/// @param u_matrix Input velocity matrix
/// @param d_matrix Matrix to apply advection
/// @return
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
    float *dy_new)
{
  // Trace back the trajectory given the current velocity
  const float px = ((float)x) - ((float)rdx) * timestep * u_matrix[matrix_index(x, y, dim, 0)];
  const float py = ((float)y) - ((float)rdx) * timestep * u_matrix[matrix_index(x, y, dim, 1)];

  // Given the traceback position, perform bilinear interpolation
  // using the 4 neighboring points and load into the output result
  bilinear_interpolation(px, py, d_matrix, dim, dx_new, dy_new);
}

__host__ __device__ Vector jacobi(
    const size_t x, const size_t y,
    const float *x_vector,
    const float *b_vector,
    const MatrixDim dim,
    const float alpha,
    const float beta)
{    
    Vector vN, vS, vE, vW;
    neighbors(x, y, x_vector, dim, &vN, &vS, &vE, &vW);

    const Vector vC = {
        .x = b_vector[matrix_index(x, y, dim, 0)],
        .y = b_vector[matrix_index(x, y, dim, 1)]};

    return (vN + vS + vE + vW + (vC * alpha)) * beta;
}