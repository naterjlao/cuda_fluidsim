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

//-----------------------------------------------------------------------------
/// @brief
/// @param dim Dimension Specification for Vector Field.
/// @param velocity Input VECTOR Velocity Field
/// @param div Output SCALAR Divergence Field
/// @param halfrdx Divergence Factor Constant
/// @return None.
//-----------------------------------------------------------------------------
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
    div[y * dim.x + x] = divergence(x, y, velocity, dim, halfrdx);
  }
}

__host__ __device__ float divergence(
    const size_t x, const size_t y,
    const float *data,
    const MatrixDim dim,
    const float halfrdx)
{
  Vector vN, vS, vE, vW;
  neighbors_vector(x, y, data, dim, &vN, &vS, &vE, &vW);

  return halfrdx * (vE.x - vW.x + vS.y - vN.y);
}

__global__ void kernel_jacobi(
    const MatrixDim dim,
    float *X,
    const float *B,
    const float alpha,
    const float beta)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < dim.x) && (y < dim.y))
  {
    X[y * dim.x + x] = jacobi(x, y, X, B, dim, alpha, beta);
  }
}

__host__ __device__ float jacobi(
    const size_t x, const size_t y,
    const float *X,
    const float *B,
    const MatrixDim dim,
    const float alpha,
    const float beta)
{
  float sN, sS, sE, sW;
  neighbors_scalar(x, y, X, dim, &sN, &sS, &sE, &sW);
  const float sB = B[y * dim.x + x];

  return (sN + sS + sE + sW + alpha * sB) * beta;
}

__global__ void kernel_sboundary(
    const MatrixDim dim,
    float *M,
    const float scale)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  /// @todo There is probably a more elegant way to do this
  if ((x < dim.x) && (y < dim.y))
  {
    if ((x == 0) && (y == 0)) // North-West Corner
    {
      M[y * dim.x + x] = scale * M[(y + 1) * dim.x + (x + 1)];
    }
    else if ((x == (dim.x - 1)) && (y == 0)) // North-East Corner
    {
      M[y * dim.x + x] = scale * M[(y + 1) * dim.x + (x - 1)];
    }
    else if ((x == 0) && (y == (dim.y - 1))) // South-West Corner
    {
      M[y * dim.x + x] = scale * M[(y - 1) * dim.x + (x + 1)];
    }
    else if ((x == (dim.x - 1)) && (y == (dim.y - 1))) // South-East Corner
    {
      M[y * dim.x + x] = scale * M[(y - 1) * dim.x + (x - 1)];
    }
    else if (x == 0) // West Border
    {
      M[y * dim.x + x] = scale * M[y * dim.x + (x + 1)];
    }
    else if (y == 0) // North Border
    {
      M[y * dim.x + x] = scale * M[(y + 1) * dim.x + x];
    }
    else if (x == (dim.x - 1)) // East Border
    {
      M[y * dim.x + x] = scale * M[y * dim.x + (x - 1)];
    }
    else if (y == (dim.y - 1)) // South Border
    {
      M[y * dim.x + x] = scale * M[(y - 1) * dim.x + x];
    }
  }
}

__global__ void kernel_vboundary(
    const MatrixDim dim,
    float *M,
    const float scale)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  /// @todo There is probably a more elegant way to do this
  if ((x < dim.x) && (y < dim.y))
  {
    if ((x == 0) && (y == 0)) // North-West Corner
    {
      M[matrix_index(x, y, dim, 0)] = scale * M[matrix_index(x + 1, y + 1, dim, 0)];
      M[matrix_index(x, y, dim, 1)] = scale * M[matrix_index(x + 1, y + 1, dim, 1)];
    }
    else if ((x == (dim.x - 1)) && (y == 0)) // North-East Corner
    {
      M[matrix_index(x, y, dim, 0)] = scale * M[matrix_index(x - 1, y + 1, dim, 0)];
      M[matrix_index(x, y, dim, 1)] = scale * M[matrix_index(x - 1, y + 1, dim, 1)];
    }
    else if ((x == 0) && (y == (dim.y - 1))) // South-West Corner
    {
      M[matrix_index(x, y, dim, 0)] = scale * M[matrix_index(x + 1, y - 1, dim, 0)];
      M[matrix_index(x, y, dim, 1)] = scale * M[matrix_index(x + 1, y - 1, dim, 1)];
    }
    else if ((x == (dim.x - 1)) && (y == (dim.y - 1))) // South-East Corner
    {
      M[matrix_index(x, y, dim, 0)] = scale * M[matrix_index(x - 1, y - 1, dim, 0)];
      M[matrix_index(x, y, dim, 1)] = scale * M[matrix_index(x - 1, y - 1, dim, 1)];
    }
    else if (x == 0) // West Border
    {
      M[matrix_index(x, y, dim, 0)] = scale * M[matrix_index(x + 1, y, dim, 0)];
    }
    else if (y == 0) // North Border
    {
      M[matrix_index(x, y, dim, 1)] = scale * M[matrix_index(x, y + 1, dim, 0)];
    }
    else if (x == (dim.x - 1)) // East Border
    {
      M[matrix_index(x, y, dim, 0)] = scale * M[matrix_index(x - 1, y, dim, 0)];
    }
    else if (y == (dim.y - 1)) // South Border
    {
      M[matrix_index(x, y, dim, 1)] = scale * M[matrix_index(x, y - 1, dim, 0)];
    }
  }
}
