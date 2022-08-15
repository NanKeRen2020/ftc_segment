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

// ------------------------------  辅助函数接口   -----------------
// 按坐标值大小排序找到的轮廓
inline void sort_contours(std::vector<std::vector<cv::Point>>& contours);

std::string filter_string(const std::string str1, const std::string str2);

void check_roi_valid(cv::Rect& roi, int width, int height);

cv::Mat weakAreaCheck(const cv::Mat& testImg, int blur_size = 3, double sigma = 1.0);

void change_bg_black(cv::Mat& binary_img, const cv::Rect roi = cv::Rect());

// 转变为黑底白字
void change_bg_black(cv::Mat& binary_img, float low_thr_ratio, float high_thr_ratio = 0.8);

// 光照均匀化
void compensate_uneven_light(cv::Mat &image, int blockSize = 32);

void canny_free( const cv::Mat &image, cv::Mat &edgeMap, std::pair<float, float> canny_thr, 
                 bool update_parameter = false, int gaussianSize = 5, float VMGradient = 0.8);

void binary_sauvola(const cv::Mat& inpImg, cv::Mat& resImg, int window = 5, float k = 1);

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


template<typename T1, typename T2, typename T3>
bool detect_gap(const cv::Mat& to_project, bool horizon_project, T1&& integral_width, 
                T2&& missing_thr1, T3&& missing_thr2, bool update_parameter = false, uchar col_label = 0)
{
    cv::Mat project;
    cv::reduce(to_project, project, horizon_project, cv::REDUCE_SUM, CV_32S);    
    std::vector<unsigned int> integral_project = get_local_integral(project, integral_width);
    //std::cout << "detect_gap integral_project: " << integral_project << std::endl;
    
    /*
    auto min_beg = std::min_element(integral_project.begin() + 0, min_itr - 1);
    auto min_end = std::min_element(min_itr + 1, integral_project.end() - 0);
    if ( (integral_project.end() - min_itr < 2*integral_width) ||
          ( min_itr - integral_project.begin() < 2*integral_width) )
    {
        

    }
    if (integral_project.end() - min_itr > 2*integral_width)
    */
    
    
    bool result = false;
    // 行字符/全列字符
    if (!horizon_project || (horizon_project && col_label == 9))
    {
        auto min_itr = std::min_element(integral_project.begin() + 0, integral_project.end() - 0);
        std::cout << ", min value global: " << *min_itr << ", ";
        result = ( (*min_itr) < missing_thr1 || (*min_itr) > missing_thr2 );

        if (update_parameter && !(*min_itr) && !horizon_project && integral_width < to_project.cols/3 ||
		    update_parameter && !(*min_itr) && horizon_project && integral_width < to_project.rows/2)
		{
			integral_width += 2;
		}
        if (update_parameter && (*min_itr) < missing_thr1 )
        {
            missing_thr1 = (!(*min_itr)) ? 600 : (*min_itr)*0.8;
        }
        if (update_parameter && (*min_itr) > missing_thr2 )
        {
            missing_thr2 = (*min_itr);
        }
    }
    if (horizon_project && col_label != 9)
    {
        int w = (integral_project.size() + integral_width)/3;
        auto min_itr1 = std::min_element(integral_project.begin() + 0, integral_project.begin() + w);
        auto min_itr2 = std::min_element(integral_project.begin() + w, integral_project.begin() + 2*w);
        auto min_itr3 = std::min_element(integral_project.begin() + 2*w, integral_project.end() - 0);
        std::cout << ", min value three: " << *min_itr1 << ", " << *min_itr2 << ", " << *min_itr3 << std::endl;
        int result1 = 9;
        if ( (*min_itr1) < missing_thr1 ) result1 = result1 - 2;
        if ( (*min_itr2) < missing_thr1 ) result1 = result1 - 3;
        if ( (*min_itr3) < missing_thr1 ) result1 = result1 - 4;
        result = (col_label != result1);
    }
    
    return result;
}


void adjust_angle(const cv::Mat& img, float angle, cv::Mat& affine_img);


std::vector<cv::Rect> segment_ftc(const cv::Mat& edge_to_project, int integral_length, bool horizon_project, int cost_thresh, float eps);

#endif // STD_UTILS_H