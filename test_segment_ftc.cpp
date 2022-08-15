#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "std_utils.h"
#include "libftc.h"


/*

g++ -g -o test_segment_ftc test_segment_ftc.cpp std_utils.cpp libftc.cpp -I. -fopenmp  -lpthread -lboost_system -lboost_filesystem `pkg-config --cflags --libs opencv`

*/

int main(int argc, char* argv[])
{
    cv::Mat image = cv::imread(argv[1]);
    std::cout << image.size() << ", " << image.channels() << std::endl;
    cv::Mat filtered_image = preprocess_text_area(image, 2, 5, 5, 3, 1, 0, 0);
    cv::Mat edeg_image;
    cv::Canny(filtered_image, edeg_image, 0, 30);
    cv::imshow("edeg_image", edeg_image);
    cv::waitKey();

    cv::Mat show_rows = image.clone();
    std::vector<cv::Rect> row_rois = segment_ftc(edeg_image, 8, true, 80000, 0.0001);
    std::vector<cv::Rect> row_char_rois;
    cv::Point p1, p2;
    for (int i = 0; i < row_rois.size(); ++i)
    {
        p1 = cv::Point(row_rois[i].x, row_rois[i].y + row_rois[i].height);
        p2 = cv::Point(row_rois[i].x + row_rois[i].width, row_rois[i].y + row_rois[i].height);
        if ( i < row_rois.size() - 1)
        cv::line(show_rows, p1, p2, cv::Scalar(0, 255, 0), 2);
        cv::imshow("detect_rows", show_rows);
        cv::waitKey();
        row_char_rois = segment_ftc(edeg_image(row_rois[i]), 8, false, 10000, 0.0001);
        cv::Mat show_chars = image(row_rois[i]);
        for (int j = 0; j < row_char_rois.size(); ++j)
        {
            p1 =cv::Point(row_char_rois[j].x + row_char_rois[j].width, row_char_rois[j].y);
            p2 = cv::Point(row_char_rois[j].x + row_char_rois[j].width, row_char_rois[j].y + row_char_rois[j].height);
            if ( j < row_char_rois.size() - 1)
            cv::line(show_chars, p1, p2, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("detect_row_chars", show_chars);
        cv::waitKey();
    }
    
    return 0;

}