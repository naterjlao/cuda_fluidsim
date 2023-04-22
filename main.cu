#include <opencv2/opencv.hpp>
#include <stdio.h>

#include "include/gradient.hpp"

int main()
{
    const size_t HEIGHT = 500;
    const size_t WIDTH = 500;
    unsigned int data[HEIGHT * WIDTH];
    for (size_t i = 0; i < HEIGHT * WIDTH; i++)
        data[i] = 0xFF0000FF;

    cv::Mat image = cv::Mat(HEIGHT, WIDTH, CV_8UC4, (unsigned *)data);

    printf("0x%X\n",normalized2bgr(1.0));

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);

    return 0;
}