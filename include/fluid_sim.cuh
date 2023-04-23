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
    float *dy_new);