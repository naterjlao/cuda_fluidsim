#include <math.h>
#include "include/gradient.cuh"

__global__ void kernel_gradient(
    const float *field,
    unsigned int *bgr,
    const size_t dim_x,
    const size_t dim_y)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;  
    if (y < dim_y && x < dim_x)
    {
        float r = field[(y * dim_x + x) * 2];
        float g = field[(y * dim_x + x) * 2 + 1];

        r = ((float) 0xFF) * (abs(r) > 1.0 ? 1.0 : abs(r));
        g = ((float) 0xFF) * (abs(g) > 1.0 ? 1.0 : abs(g));

        // This is actually little-endian,
        // so octets are arranged like: 0xAARRGGBB
        bgr[y * dim_x + x] = 0xFF000000;
        bgr[y * dim_x + x] |= 0x00FF0000 & (((unsigned char) r) << 16);
        bgr[y * dim_x + x] |= 0x0000FF00 & (((unsigned char) g) << 8);
    }
}

