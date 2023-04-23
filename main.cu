#include <opencv2/opencv.hpp>
#include <stdio.h>

#include "include/gradient.hpp"

int main()
{
    const size_t HEIGHT = 800;
    const size_t WIDTH = 800;
    const size_t BUFFER_LEN = HEIGHT * WIDTH;
    const size_t BUFFER_SIZE = sizeof(unsigned int) * BUFFER_LEN;
    const size_t BLOCK_SIZE = 1024;
    const size_t BLOCK_NUM = BUFFER_LEN / BLOCK_SIZE + ((BUFFER_LEN % BLOCK_SIZE > 0) ? 1 : 0);

    unsigned int *data;
    unsigned int *device_data;

    cudaMallocHost(&data, BUFFER_SIZE);
    cudaMalloc(&device_data, BUFFER_SIZE);
    memset(data, 0, BUFFER_SIZE);

    cv::Mat image;

    while (true)
    {
        cudaMemcpy(device_data, data, BUFFER_SIZE, cudaMemcpyHostToDevice);
        test<<<BLOCK_NUM, BLOCK_SIZE>>>(device_data);
        cudaMemcpy(data, device_data, BUFFER_SIZE, cudaMemcpyDeviceToHost);

        // cudaDeviceSynchronize();

        image = cv::Mat(HEIGHT, WIDTH, CV_8UC4, (unsigned *)data);

        // printf("0x%X\n",normalized2bgr(1.0));
        //printf("0x%X\n", data[1]);

        cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display Image", image);
        cv::waitKey(1);
    }

    cudaFree(data);
    cudaFree(device_data);
    return 0;
}