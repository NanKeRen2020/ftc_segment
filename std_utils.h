#ifndef STD_UTILS_H
#define STD_UTILS_H

#include <opencv2/opencv.hpp>
#include <numeric>

enum class SMOOTH_METHOD : int
{
    NO_SMOOTH                   = 0,
    NORMALIZE_BLUR              = 1,
	GAUSSIAN_BLUR               = 2,
};

enum class DENOISE_METHOD : int
{
    NO_DENOISE                  = 0,
    MEDIAN_BLUR                 = 1,
	BILATERAL_FILTER            = 2,
	FAST_NLM_DENOISE            = 3,
};


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


// 光照均匀化
void compensate_uneven_light(cv::Mat &image, int blockSize = 32);

void canny_free( const cv::Mat &image, cv::Mat &edgeMap, std::pair<float, float> canny_thr, 
                 bool update_parameter = false, int gaussianSize = 5, float VMGradient = 0.8);

// 定位投影数组的边界
std::pair<unsigned int, unsigned int> find_boundary(const std::vector<unsigned int>& project_vector, unsigned int boundary_thr1, 
                                                    unsigned int boundary_thr2, const std::vector<unsigned int>::iterator start);

// 投影矩阵局部积分
std::vector<unsigned int> get_local_integral(const cv::Mat& project, int integral_width = 0);



// 找数组最小值，定位字符间隙
void find_minimums(const std::vector<unsigned int>::iterator start, const std::vector<unsigned int>::iterator begin, const std::vector<unsigned int>::iterator end, 
                   std::vector<int>& minimums, int stop_width, bool stop_width_halve = false, int gap_max_value = 0);

// 对比度拉伸
cv::Mat adjust_contrast(const cv::Mat& gray, float gamma = 1.0);

// 图像平滑
cv::Mat smooth(const cv::Mat& gray, const SMOOTH_METHOD s_method = SMOOTH_METHOD::NO_SMOOTH, int size = 5, float sigma = 3);

// 图像去噪
cv::Mat denoise(const cv::Mat& gray, const DENOISE_METHOD d_method = DENOISE_METHOD::NO_DENOISE, int size = 3);

cv::Mat preprocess_text_area(const cv::Mat& text_area, uchar text_denoise, uchar text_denoise_size, uchar text_smooth, uchar text_smooth_size,
                             float  text_smooth_sigma, uchar compensate_size, float text_alpha);

std::vector<cv::Rect> segment_ftc(const cv::Mat& edge_to_project, int integral_length, bool horizon_project, int cost_thresh, float eps);

#endif // STD_UTILS_H