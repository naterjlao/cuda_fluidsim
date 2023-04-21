#include <opencv2/opencv.hpp>
#include <stdio.h>
int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], 1);
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }

    // Put Text
    cv::Point text_coordinates(50,50);
    cv::Scalar text_color(0,0,0);
    cv::putText(image, "test", text_coordinates, cv::FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKey(0);
    return 0;
}
