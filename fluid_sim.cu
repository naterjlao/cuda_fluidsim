#include "include/fluid_sim.cuh"

enum VECTOR_COMPONENT
{
  X_COMPONENT = 0,
  Y_COMPONENT = 1,
  N_DIMENIONS = 2
};

/// @brief Returns the array index of a matrix vector array given dimenion coordinates
/// @param dim_x
/// @param dim_y
/// @param x
/// @param y
/// @param component
/// @return
static __device__ bool vector_component_idx(
    const unsigned int dim_x,
    const unsigned int dim_y,
    const unsigned int x,
    const unsigned int y,
    const VECTOR_COMPONENT component,
    size_t *vector_idx)
{
  bool retval = false;
  if (x < dim_x && y < dim_y)
  {
    *vector_idx = (y * dim_x + x) * 2 + component;
    retval = true;
  }
  return retval;
}

/// @brief Returns the value of a matrix vector array given dimension coordinates
/// @param dim_x
/// @param dim_y
/// @param x
/// @param y
/// @param component
/// @param vector
/// @return
static __device__ float vector_component(
    const unsigned int dim_x,
    const unsigned int dim_y,
    const unsigned int x,
    const unsigned int y,
    const VECTOR_COMPONENT component,
    const float *vector)
{
  float retval = 0.0;
  size_t vector_idx;
  if (vector_component_idx(dim_x, dim_y, x, y, component, &vector_idx))
  {
    retval = vector[vector_idx];
  }
  return retval;
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
    const size_t dim_x,
    const size_t dim_y,
    const size_t coord_x,
    const size_t coord_y,
    const size_t rdx,
    const float timestep,
    const float dissipation,
    const float *u_matrix,
    const float *d_matrix,
    float *dx_new,
    float *dy_new)
{
  // Trace back the trajectory given the current velocity
  const size_t pos_x = (size_t)(((float)coord_x) - ((float) rdx) * timestep
    * vector_component(dim_x, dim_y, coord_x, coord_y, X_COMPONENT, u_matrix));
  const size_t pos_y = (size_t)(((float)coord_y) - ((float) rdx) * timestep
    * vector_component(dim_x, dim_y, coord_x, coord_y, Y_COMPONENT, u_matrix));

  // Given the traceback position, perform bilinear interpolation of the
  // 4 neighboring points and load into the output matrix
  float v_n, v_s, v_e, v_w;
  
  v_n = vector_component(dim_x, dim_y, pos_x, pos_y - 1, X_COMPONENT, d_matrix);
  v_s = vector_component(dim_x, dim_y, pos_x, pos_y + 1, X_COMPONENT, d_matrix);
  v_e = vector_component(dim_x, dim_y, pos_x + 1, pos_y, X_COMPONENT, d_matrix);
  v_w = vector_component(dim_x, dim_y, pos_x - 1, pos_y, X_COMPONENT, d_matrix);
  *dx_new = dissipation * ((v_n + v_s + v_e + v_w) / 4.0);

  v_n = vector_component(dim_x, dim_y, pos_x, pos_y - 1, Y_COMPONENT, d_matrix);
  v_s = vector_component(dim_x, dim_y, pos_x, pos_y + 1, Y_COMPONENT, d_matrix);
  v_e = vector_component(dim_x, dim_y, pos_x + 1, pos_y, Y_COMPONENT, d_matrix);
  v_w = vector_component(dim_x, dim_y, pos_x - 1, pos_y, Y_COMPONENT, d_matrix);
  *dy_new = dissipation * ((v_n + v_s + v_e + v_w) / 4.0);
}