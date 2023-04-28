#include "include/fluid_sim.cuh"

__global__ void kernel_advect(
    const MatrixDim dim,
    const float *input_data,
    float *output_data,
    const float rdx,
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
    float *dy_new)
{
  // Trace back the trajectory given the current velocity
  const float px = ((float)x) - rdx * timestep * u_matrix[matrix_index(x, y, dim, 0)];
  const float py = ((float)y) - rdx * timestep * u_matrix[matrix_index(x, y, dim, 1)];

  // Given the traceback position, perform bilinear interpolation
  // using the 4 neighboring points and load into the output result
  bilinear_interpolation(px, py, d_matrix, dim, dx_new, dy_new);
}

__global__ void kernel_divergence(
    const MatrixDim dim,
    const float *velocity,
    float *div,
    const float halfrdx)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < dim.x) && (y < dim.y))
  {
    Vector v_div = divergence(x, y, velocity, dim, halfrdx);
    div[matrix_index(x, y, dim, 0)] = v_div.x;
    div[matrix_index(x, y, dim, 1)] = v_div.y;
  }
}
__host__ __device__ Vector divergence(
    const size_t x, const size_t y,
    const float *data,
    const MatrixDim dim,
    const float halfrdx)
{
  Vector vN, vS, vE, vW;
  neighbors(x, y, data, dim, &vN, &vS, &vE, &vW);

  Vector div;
  div.x = halfrdx * (vE.x - vW.x);
  div.y = halfrdx * (vS.y - vN.y);

  return div;
}

__global__ void kernel_jacobi(
    const MatrixDim dim,
    const float *X,
    const float *B,
    float *X_new,
    const float alpha,
    const float beta,
    const size_t iterations)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < dim.x) && (y < dim.y))
  {
    for (size_t iter = 0; iter < iterations; iter++)
    {
      Vector x_new = jacobi(x, y, X, B, dim, alpha, beta);
      X_new[matrix_index(x, y, dim, 0)] = x_new.x;
      X_new[matrix_index(x, y, dim, 1)] = x_new.x;
    }
  }
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

  const Vector vB = {
      .x = b_vector[matrix_index(x, y, dim, 0)],
      .y = b_vector[matrix_index(x, y, dim, 1)]};

  return (vN + vS + vE + vW + (vB * alpha)) * beta;
}