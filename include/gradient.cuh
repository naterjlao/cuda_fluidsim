__global__ void kernel_gradient(
    const float *field, 
    unsigned int *bgr,
    const size_t dim_x,
    const size_t dim_y);