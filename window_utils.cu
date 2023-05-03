#include <math.h>
#include <curand.h>
#include "include/window_utils.cuh"
#include "include/fluid_utils.cuh"

__global__ void kernel_vfield2bgr(
    const float *field,
    unsigned int *bgr,
    const MatrixDim dim)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dim.x && y < dim.y)
    {
        float r = field[matrix_index(x, y, dim, 0)];
        float b = field[matrix_index(x, y, dim, 1)];

        r = ((float)0xFF) * (abs(r) > 1.0 ? 1.0 : abs(r));
        b = ((float)0xFF) * (abs(b) > 1.0 ? 1.0 : abs(b));

        // This is actually little-endian,
        // so octets are arranged like: 0xAARRGGBB
        bgr[y * dim.x + x] = 0xFF000000;
        bgr[y * dim.x + x] |= 0x00FF0000 & (((unsigned char)r) << 16);
        bgr[y * dim.x + x] |= 0x000000FF & (((unsigned char)b) << 0);
    }
}

__global__ void kernel_sfield2bgr(
    const float *field,
    unsigned int *bgr,
    const MatrixDim dim)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dim.x && y < dim.y)
    {
        float v = field[y * dim.x + x];
        const float r = (v > 0.0) ? (((float)0xFF) * abs(v)) : 0.0;
        const float b = (v < 0.0) ? (((float)0xFF) * abs(v)) : 0.0;

        // This is actually little-endian,
        // so octets are arranged like: 0xAARRGGBB
        bgr[y * dim.x + x] = 0xFF000000;
        bgr[y * dim.x + x] |= 0x00FF0000 & (((unsigned char)r) << 16);
        bgr[y * dim.x + x] |= 0x000000FF & (((unsigned char)b) << 0);
    }
}

__global__ void kernel_pulse(
    const size_t epicenter_x, const size_t epicenter_y,
    float *field, const MatrixDim dim)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x < dim.x && y < dim.y) && !((x == epicenter_x) && (y == epicenter_y)))
    {
        // field[matrix_index(x,y,dim,0)] = (-1.0) *((float) (x - epicenter_x)) / ((float ) epicenter_x);
        // field[matrix_index(x,y,dim,0)] = -1.0;
        // field[matrix_index(x,y,dim,0)] += (x == epicenter_x) ? 0.0 : 1.0 / (((float) (x)) - ((float) epicenter_x));
        // field[matrix_index(x,y,dim,1)] += (y == epicenter_y) ? 0.0 : 1.0 / (((float) (y)) - ((float) epicenter_y));
        // field[matrix_index(x,y,dim,1)] = 2.0 ;

        const float f_x = ((float)x) - ((float)epicenter_x);
        const float f_y = ((float)y) - ((float)epicenter_y);

        const float theta = atan(f_y / f_x);
        const float radial = sqrt((f_x * f_x) + (f_y * f_y));
        const float max = sqrt((float)((dim.x > dim.y) ? dim.x : dim.y));

        field[matrix_index(x, y, dim, 0)] += cos(theta) * (max / (max + radial)) * ((x < epicenter_x) ? -1.0 : 1.0); /** @note awkward hotfix for coordinate issues*/
        field[matrix_index(x, y, dim, 1)] += sin(theta) * (max / (max + radial)) * ((x < epicenter_x) ? -1.0 : 1.0);
    }
}
