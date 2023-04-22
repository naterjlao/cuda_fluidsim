#include "include/gradient.hpp"
/*
FF0000 -> BLUE
FFFF00
00FF00
00FFFF
0000FF -> RED
FF00FF -> VIOLET
*/

__host__ unsigned int normalized2bgr(const float data)
{
    return (unsigned int) 8.0;
}

__global__ void test(unsigned int *data)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    data[thread_idx] = 0x0000FFFF;
}