//-----------------------------------------------------------------------------
/// @file fluid_sim.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief CUDA Fluid Simulation Computartion Functions
//-----------------------------------------------------------------------------
#include "include/fluid_sim.cuh"

//-----------------------------------------------------------------------------
/// @brief Computes the Advection Computation
/// @param dim Defines the working space dimension bounds.
/// @param input_data Input Vector Field Matrix
/// @param output_data Output Vector Field Matrix
/// @param rdx Divergence Factor Constant
/// @param timestep Timestep Factor
/// @param dissipation Dissipation Factor
/// @return None.
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
/// @brief Performs Advection Computation at a specified coordinate.
/// @param dim Defines the working space dimension bounds.
/// @param x x coordinate
/// @param y y coordinate
/// @param rdx Divergence Factor Constant
/// @param timestep Timestep Factor
/// @param dissipation Dissipation Factor
/// @param u_matrix Input Matrix
/// @param d_matrix Output Matrix
/// @param dx_new computed x vector component
/// @param dy_new computed y vector component
/// @return None.
//-----------------------------------------------------------------------------
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
/// @brief Computes Divergence Computation
/// @param dim Defines the working space dimension bounds.
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
  if ((x > 0) && (y > 0) && (x < dim.x - 1) && (y < dim.y - 1))
  {
    div[y * dim.x + x] = divergence(x, y, velocity, dim, halfrdx);
  }
}

//-----------------------------------------------------------------------------
/// @brief Performs Advection Computation at a specified coordinate.
/// @param x x coordinate
/// @param y y coordinate
/// @param data Input velocity vector field matrix
/// @param dim Defines the working space dimension bounds.
/// @param halfrdx rdx factor constant
/// @return the computed divergence at a given coordinate.
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
/// @brief Computes a Jacobi solution step computation.
/// @param dim Defines the working space dimension bounds.
/// @param X X vector
/// @param B B vector
/// @param alpha alpha constant factor
/// @param beta beta constant factor
/// @return None.
//-----------------------------------------------------------------------------
__global__ void kernel_jacobi(
    const MatrixDim dim,
    float *X,
    const float *B,
    const float alpha,
    const float beta)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x > 0) && (y > 0) && (x < dim.x - 1) && (y < dim.y - 1))
  {
    X[y * dim.x + x] = jacobi(x, y, X, B, dim, alpha, beta);
  }
}

//-----------------------------------------------------------------------------
/// @brief Computes a Jacobi solution step computation at a specific coordinate.
/// @param x x coordinate
/// @param y y coordinate
/// @param X X vector
/// @param B B vector
/// @param dim Defines the working space dimension bounds.
/// @param alpha alpha constant factor
/// @param beta beta constant factor
/// @return The jacobi solution at a given point.
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
/// @brief Computes boundary values.
/// @param dim Defines the working space dimension bounds.
/// @param M Scalar Matrix field
/// @param scale boundary scale factor
/// @return None.
//-----------------------------------------------------------------------------
__global__ void kernel_sboundary(
    const MatrixDim dim,
    float *M,
    const float scale)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < dim.x) && (y < dim.y))
  {
    /// @todo There is probably a more elegant way to do this
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

//-----------------------------------------------------------------------------
/// @brief Computes boundary values.
/// @param dim Defines the working space dimension bounds.
/// @param M Vector Matrix field.
/// @param scale boundary scale factor
/// @return None.
//-----------------------------------------------------------------------------
__global__ void kernel_vboundary(
    const MatrixDim dim,
    float *M,
    const float scale)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < dim.x) && (y < dim.y))
  {
    /// @todo There is probably a more elegant way to do this
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

//-----------------------------------------------------------------------------
/// @brief Subtracts the Pressure Gradient scalars from the Velocity Vector Matrix.
/// @param dim Defines the working space dimension bounds.
/// @param P Pressure Scalar Field Matrix
/// @param V Velocity Vector Field Matrix.
/// @param halfrdx rdx factor
/// @return None.
//-----------------------------------------------------------------------------
__global__ void kernel_gradient(
    const MatrixDim dim,
    const float *P,
    float *V,
    const float halfrdx)
{
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x > 0) && (y > 0) && (x < dim.x - 1) && (y < dim.y - 1))
  {
    float pN, pS, pE, pW;
    neighbors_scalar(x, y, P, dim, &pN, &pS, &pE, &pW);

    const float gradX = halfrdx * (pW - pS);
    const float gradY = halfrdx * (pS - pN);

    V[matrix_index(x, y, dim, 0)] -= gradX;
    V[matrix_index(x, y, dim, 1)] -= gradY;
  }
}
