#include <math.h>
#include "include/gradient.cuh"
#include "include/fluid_utils.cuh"

__global__ void kernel_gradient(
    const float *field,
    unsigned int *bgr,
    const MatrixDim dim)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;  
    if (x < dim.x && y < dim.y)
    {
        float r = field[matrix_index(x,y,dim,0)];
        float b = field[matrix_index(x,y,dim,1)];

        r = ((float) 0xFF) * (abs(r) > 1.0 ? 1.0 : abs(r));
        b = ((float) 0xFF) * (abs(b) > 1.0 ? 1.0 : abs(b));

        // This is actually little-endian,
        // so octets are arranged like: 0xAARRGGBB
        bgr[y * dim.x + x] = 0xFF000000;
        bgr[y * dim.x + x] |= 0x00FF0000 & (((unsigned char) r) << 16);
        bgr[y * dim.x + x] |= 0x000000FF & (((unsigned char) b) << 0);
    }
}

