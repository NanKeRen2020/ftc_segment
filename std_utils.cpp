
#include <numeric>
#include <vector>
#include <string>
#include <limits>

#include <boost/filesystem.hpp>
#include "std_utils.h"
#include "libftc.h"


inline void sort_contours_coordinate(std::vector<std::vector<cv::Point>>& contours)
{
    std::sort(contours.begin(), contours.end(), 
              [](const std::vector<cv::Point> con1, const std::vector<cv::Point> con2)
                { return cv::boundingRect(con1).x < cv::boundingRect(con2).x;});   
}

inline void sort_contours_area(std::vector<std::vector<cv::Point>>& contours)
{
    std::sort(contours.begin(), contours.end(), 
              [](const std::vector<cv::Point> con1, const std::vector<cv::Point> con2)
                { return cv::contourArea(con1) > cv::contourArea(con2);});   
}

cv::Mat denoise(const cv::Mat& gray, const DENOISE_METHOD d_method, int size)
{
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat denoised = gray.clone();
    if (d_method == DENOISE_METHOD::MEDIAN_BLUR)
    {
       cv::medianBlur(gray, denoised, size);
    }
    if (d_method == DENOISE_METHOD::BILATERAL_FILTER)
    {
       cv::bilateralFilter(gray, denoised, size, 16, 4);
    }
    if (d_method == DENOISE_METHOD::FAST_NLM_DENOISE)
    {
       cv::fastNlMeansDenoising(gray, denoised, size, 5, 3);
    }
    auto end = std::chrono::high_resolution_clock::now();                
    std::chrono::duration<double> diff = end - start;
    std::cout << "===>image_process_denoise_time: " << diff.count() << std::endl;
    return denoised;
}

cv::Mat smooth(const cv::Mat& gray, const SMOOTH_METHOD s_method, int size, float sigma)
{
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat smoothed = gray.clone();
    if (s_method == SMOOTH_METHOD::NORMALIZE_BLUR)
    {
       cv::blur(gray, smoothed, cv::Size(size, size));
    }
    if (s_method == SMOOTH_METHOD::GAUSSIAN_BLUR)
    {
       cv::GaussianBlur(gray, smoothed, cv::Size(size, size), sigma);
    }
    auto end = std::chrono::high_resolution_clock::now();                
    std::chrono::duration<double> diff = end - start;
    std::cout << "===>image_process_smooth_time: " << diff.count() << std::endl;
    return smoothed;
}

cv::Mat adjust_contrast(const cv::Mat& gray, float gamma)
{
    cv::Mat contrast = gray.clone();
    cv::normalize(contrast, contrast, 0, 255, cv::NORM_MINMAX);
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for(int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    std::cout << "adjust_contrast: " << contrast.size() << ", " << contrast.channels() << ", "
              << lookUpTable.size() << ", " << lookUpTable.channels() << std::endl;
    cv::LUT(contrast, lookUpTable, contrast);   
    return contrast;
}


cv::Mat weakAreaCheck(const cv::Mat& testImg, int blur_size, double sigma) 
{

	cv::Mat result = cv::Mat::zeros(testImg.rows, testImg.cols, CV_8UC1);
	cv::Mat grayscale = testImg.clone();
    cv::GaussianBlur(grayscale, grayscale, cv::Size(blur_size, blur_size), sigma); 
	cv::Mat gradintegral = cv::Mat::zeros(grayscale.rows, grayscale.cols, CV_32F);
	cv::Mat gradintegral2 = cv::Mat::zeros(grayscale.rows, grayscale.cols, CV_32F);
	cv::integral(grayscale, gradintegral, CV_32F); 
	float graymean = 0;
	int length = 4;
	int value = 1;
	float rate = 1.250;
	for (int i = length; i < grayscale.rows - length; i++) {
		for (int j = length; j < grayscale.cols - length; j++) {
			
				graymean = gradintegral.at<float>(i + length, j + length) + 
                gradintegral.at<float>(i - length, j - length) - gradintegral.at<float>(i - length, j + length) - 
                gradintegral.at<float>(i + length, j - length);
				graymean = graymean / ((length *2 + 1)*(length*2 + 1));
				if (grayscale.at<uchar>(i, j) < rate*graymean) {
					value = rate*graymean - grayscale.at<uchar>(i, j);
					value = value * 60;
					if (value > 255) {
						value = 254;
					}
					result.at<uchar>(i, j) = value;
				}
			
		}
	}
    return result;
}


void unevenLightCompensate(cv::Mat &image, int blockSize)
{
	if (image.channels() == 3) cv::cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	cv::Mat blockImage;
	blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			cv::Mat imageROI = image(cv::Range(rowmin, rowmax), cv::Range(colmin, colmax));
			double temaver = cv::mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	cv::Mat blockImage2;
	cv::resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), cv::INTER_CUBIC);
	cv::Mat image2;
	image.convertTo(image2, CV_32FC1);
	cv::Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}

// 对灰度图像进行光照均匀化
void compensate_uneven_light(cv::Mat &gray_img, int blockSize)
{
	if (gray_img.channels() == 3) 
    cv::cvtColor(gray_img, gray_img, cv::COLOR_BGR2GRAY);

	double average = mean(gray_img)[0];
	int rows_new = ceil(double(gray_img.rows) / double(blockSize));
	int cols_new = ceil(double(gray_img.cols) / double(blockSize));
	cv::Mat blockImage;
	blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > gray_img.rows) rowmax = gray_img.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > gray_img.cols) colmax = gray_img.cols;
			cv::Mat imageROI = gray_img(cv::Range(rowmin, rowmax), cv::Range(colmin, colmax));
			double temaver = cv::mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	cv::Mat blockImage2;
	cv::resize(blockImage, blockImage2, gray_img.size(), (0, 0), (0, 0), cv::INTER_CUBIC);
	cv::Mat image2;
	gray_img.convertTo(image2, CV_32FC1);
	cv::Mat dst = image2 - blockImage2;
	dst.convertTo(gray_img, CV_8UC1);
}

void check_roi_valid(cv::Rect& roi, int width, int height)
{
    /*
    roi.x = (roi.x < 0) ? 0 : roi.x;
    roi.y = (roi.y < 0) ? 0 : roi.y;
    */
    if (roi.x < 0)
    {
        roi.width += roi.x;
        roi.x = 0;
    }
    if (roi.y < 0)
    {
        roi.height += roi.y;
        roi.y = 0;
    }
    if (roi.x + roi.width > width)
    {
        std::cout << roi.x << ", " << roi.width << ", " << width << ", ";
        roi.width = width - roi.x - 1;
    }
    if (roi.y + roi.height > height)
    {
        std::cout << roi.y << ", " << roi.height << ", " << height;
        roi.height = height - roi.y - 1;
    }
    std::cout << std::endl;
}

void change_bg_black(cv::Mat& binary_img, const cv::Rect roi)
{
    cv::Mat bin_roi = binary_img;
    if (roi.width > 0 && roi.height > 0)  bin_roi = bin_roi(roi);
    cv::Mat zero_mask = cv::Mat::zeros(3*bin_roi.rows/4, 3*bin_roi.cols/4, CV_8UC1);
    cv::Mat substract = bin_roi.clone();
    zero_mask.copyTo(substract(cv::Rect(bin_roi.cols/8, bin_roi.rows/8, 3*bin_roi.cols/4, 3*bin_roi.rows/4)));
    /*
    if (show_image)
    {
        show_mat = substract;
        show_img("substract", show_mat);
    } 
    */   
    if (cv::sum(substract)[0] > 255*bin_roi.rows*bin_roi.cols/4)
    { 
        std::cout << "substract sum: " << cv::sum(substract)[0] << std::endl;
        binary_img = ~binary_img;
    }
}


// 统计二值图像像素值 将图像改为黑色背景白色前景图像
void change_bg_black(cv::Mat& binary_img, float low_thr_ratio, float high_thr_ratio)
{
    cv::Mat project;
   
    // 沿短边投影
    int width, height;
    if (binary_img.cols >= binary_img.rows)
    {
        cv::reduce(binary_img, project, 0, cv::REDUCE_SUM, CV_32S);
        width = binary_img.cols;
        height = binary_img.rows;
    }
    else
    {
        cv::reduce(binary_img, project, 1, cv::REDUCE_SUM, CV_32S);
        width = binary_img.rows;
        height = binary_img.cols;
    }

    int low = 0;
    //int high = 0;
    int low_thr = low_thr_ratio * height;
    //int high_thr = high_thr_ratio * height;
    for (auto it = project.begin<int>(); it != project.end<int>(); ++it)
    {
        /*
        if (*it > high_thr)
           ++high;
        */
        if (*it < low_thr)
           ++low;
    }

    if ((width - low) > low)
    {
        binary_img = ~binary_img;
    }
    
}

// 计算mat数组的局部积分
std::vector<unsigned int> get_local_integral(const cv::Mat& project, int integral_width)
{
    std::vector<unsigned int> local_integral(project.begin<int>(), project.end<int>());
    if (integral_width > 0)
    {
        local_integral.clear();
        for (auto it = project.begin<int>(); it != project.end<int>() - integral_width; ++it)
        {
            local_integral.push_back( std::accumulate(it, it + integral_width, 0) );
        }
    }

    return local_integral;
    
}


void canny_free( const cv::Mat &image, cv::Mat &edgeMap, std::pair<float, float> canny_thr, 
                 bool update_parameter, int gaussianSize, float VMGradient)
{
    int apertureSize=3;

    if (update_parameter && !canny_thr.second)
    {
        int i,j,m,n;
        int grayLevels=255;
        float gNoise=1.3333;  //1.3333 
        float thGradientLow=gNoise;

        int cols = image.cols;
        int rows = image.rows;
        int cols_1=cols-1;
        int rows_1=rows-1;

        //meaningful Length
        int thMeaningfulLength=int(2.0*log((float)rows*cols)/log(8.0)+0.5);
        float thAngle=2*atan(2.0/float(thMeaningfulLength));

        //get gray image
        cv::Mat grayImage;
        int aa=image.channels();

        if ( image.channels() == 1 )
            grayImage = image;
        else
            cv::cvtColor(image, grayImage, CV_BGR2GRAY);
        //cv::normalize(image, image, 255, 0);

        //gaussian filter
        cv::Mat filteredImage;
        cv::GaussianBlur(grayImage, filteredImage, cv::Size(gaussianSize,gaussianSize), 3.0);
        grayImage.release();

        //get gradient map and orientation map
        cv::Mat gradientMap = cv::Mat::zeros(filteredImage.rows,filteredImage.cols,CV_32FC1);
        cv::Mat dx(filteredImage.rows, filteredImage.cols, CV_16S, cv::Scalar(0));
        cv::Mat dy(filteredImage.rows, filteredImage.cols, CV_16S, cv::Scalar(0));

        
        cv::Sobel(filteredImage, dx, CV_16S, 1, 0, apertureSize, 1, 0, cv::BORDER_REPLICATE);
        cv::Sobel(filteredImage, dy, CV_16S, 0, 1, apertureSize, 1, 0, cv::BORDER_REPLICATE);

        //calculate gradient and orientation
        int totalNum=0;
        int times=8;
        std::vector<int> histogram(times*grayLevels,0);
        for (i=0;i<rows;++i)
        {
            float *ptrG=gradientMap.ptr<float>(i);
            short *ptrX=dx.ptr<short>(i);
            short *ptrY=dy.ptr<short>(i);
            for (j=0;j<cols;++j)
            {
                float gx=ptrX[j];
                float gy=ptrY[j];

                ptrG[j]=abs(gx)+abs(gy);
                if (ptrG[j]>thGradientLow)
                {
                    histogram[int(ptrG[j]+0.5)]++;
                    totalNum++;
                }
                else
                    ptrG[j]=0.0;
            }
        }

        //gradient statistic
        float N2=0;
        for (i=0;i<histogram.size();++i)
        {
            if (histogram[i])
                N2+=(float)histogram[i]*(histogram[i]-1);
        }
        float pMax=1.0/exp((log(N2)/thMeaningfulLength));
        float pMin=1.0/exp((log(N2)/sqrt((float)cols*rows)));

        std::vector<float> greaterThan(times*grayLevels,0);
        int count=0;
        for (i=times*grayLevels-1;i>=0;--i)
        {
            count+=histogram[i];
            float probabilityGreater=float(count)/float(totalNum);
            greaterThan[i]=probabilityGreater;
        }
        count=0;

        //get two gradient thresholds
        int thGradientHigh=0;
        for (i=times*grayLevels-1;i>=0;--i)
        {
            if (greaterThan[i]>pMax)
            {
                thGradientHigh=i;
                break;
            }
        }
        for (i=times*grayLevels-1;i>=0;--i)
        {
            if (greaterThan[i]>pMin)
            {
                thGradientLow=i;
                break;
            }
        }
        if (thGradientLow<gNoise) thGradientLow=gNoise;

        //cout<<thGradientLow<<"  "<<thGradientHigh<<endl;

        //convert probabilistic meaningful to visual meaningful
        thGradientHigh=sqrt(thGradientHigh*VMGradient);   

        canny_thr = std::pair<int, int>(thGradientLow, thGradientHigh);  
    }
    
	cv::Canny(image, edgeMap, canny_thr.first, canny_thr.second, apertureSize);

}


std::vector<cv::Rect> detect_text_rois(cv::Mat edge_img, const cv::Size& close_size, const cv::Size& open_size, int text_rois_size)
{
    
    cv::Mat text_close, text_open;
    cv::Mat element = cv::getStructuringElement( 0, close_size);
    cv::morphologyEx(edge_img, text_close, CV_MOP_CLOSE, element);
    element = cv::getStructuringElement( 0, open_size);
    cv::morphologyEx(text_close, text_open, CV_MOP_OPEN, element); 
    //cv::imwrite("/opt/history/temp_images/text_open.png", text_open);  
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(text_open, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); 
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> con1, const std::vector<cv::Point> con2)
                    { return cv::contourArea(con1) > cv::contourArea(con2); });  
    if (contours.size() > text_rois_size)  contours.erase(contours.begin() + text_rois_size - 1, contours.end());
    std::vector<cv::Rect> text_rois;
    for (auto con: contours)
    {
        text_rois.push_back(cv::boundingRect(con));
    }
    std::sort(text_rois.begin(), text_rois.end(), [](const cv::Rect& roi1, const cv::Rect& roi2)
                    { return roi1.x < roi2.x; }); 
    return text_rois;
}


void adjust_angle(const cv::Mat& img, float angle, cv::Mat& affine_img)
{
    cv::Mat warp_mat = getRotationMatrix2D(cv::Point(img.cols/2, img.rows/2), angle, 1.0);
    cv::warpAffine(img, affine_img, warp_mat, img.size());
}


std::pair<unsigned int, unsigned int> find_boundary(const std::vector<unsigned int>& project_vector, unsigned int boundary_thr1, unsigned int boundary_thr2, 
                                                    const std::vector<unsigned int>::iterator start)
{
    unsigned int boundary1 = 0;
    unsigned int boundary2 = 0;

    for (std::vector<unsigned int>::iterator it = start; it != project_vector.cend(); ++it)
    {
        if (boundary1 == 0 && (*it > boundary_thr1))
        {
            boundary1 = it - start;
        }

        if ((boundary2 == 0) && ( *( project_vector.cend() - (it -  start) - 1 ) > boundary_thr2) )
        {
            boundary2 = it - start;

        }
        
        if ((boundary1 != 0) && (boundary2 != 0))
        {
            break;
        }
       
        
    }

    return std::pair<int, int>(boundary1, boundary2);
}


cv::Mat preprocess_text_area(const cv::Mat& text_area, uchar text_denoise, uchar text_denoise_size, uchar text_smooth, uchar text_smooth_size,
                             float text_smooth_sigma, uchar compensate_size, float text_alpha)
{
    cv::Mat text_filtered = text_area.clone();

    if ( text_denoise )
    {
        text_filtered = denoise(text_filtered, static_cast<DENOISE_METHOD>(text_denoise), text_denoise_size);
    }

    if (compensate_size)
    {
        compensate_uneven_light(text_filtered, compensate_size);
    }
    
    text_filtered = smooth(text_filtered, static_cast<SMOOTH_METHOD>(text_smooth), 
                           text_smooth_size, text_smooth_sigma);

    if (text_alpha)
    {
       text_filtered = text_filtered * text_alpha;
    }

    return text_filtered;
    
}


void find_minimums(const std::vector<unsigned int>::iterator start, const std::vector<unsigned>::iterator begin, 
                   const std::vector<unsigned>::iterator end, std::vector<int>& minimums, int stop_width, 
                   bool stop_width_halve, int gap_max_value)
{
    auto it = std::min_element(begin, end);
    int loc = static_cast<int>(it - start);
    if (gap_max_value && *it < gap_max_value)
    {
        minimums.push_back(loc);
    }
    else if (!gap_max_value)
    {
        minimums.push_back(loc);
    }
    else
    {
        /* code */
    }
    

    int diff = stop_width;
    // 停止宽度减半
    if (stop_width_halve)
    {
        diff = stop_width/2;
    }
    auto diff_begin = it  - begin;
    auto diff_end = end - it;
    if ( diff_begin > diff )
    {
        find_minimums(start, begin, it - 3, minimums, stop_width, stop_width_halve, gap_max_value);
    }
    if ( diff_end > diff )
    {
        find_minimums(start, it + 3, end, minimums, stop_width, stop_width_halve, gap_max_value);
    }
}



std::vector<cv::Rect> segment_ftc(const cv::Mat& edge_to_project, int integral_length, bool horizon_project, int cost_thresh, float eps)
{
    cv::Mat project;
    cv::reduce(edge_to_project, project, horizon_project, cv::REDUCE_SUM, CV_32S); 
    std::vector<unsigned int> integral_project = get_local_integral(project, integral_length);
    std::cout << "find_gap_location integral_project size: " << integral_project.size() << ", " << integral_project << std::endl;

    std::vector<int> seg_ftc;
    std::vector<float> iproj(integral_project.begin(), integral_project.end());
    seg_ftc = FTCsegmentation(iproj.data(), iproj.size(), cost_thresh, eps, 0); 
    std::cout << "seg_ftc: " << seg_ftc << std::endl;
    
    std::vector<cv::Rect> segment_rois;
    int h, pre;
    h = pre = 0;
    cv::Rect segment_roi;
    if (seg_ftc.size() > 1)
    {
        seg_ftc.erase(seg_ftc.begin());
        seg_ftc[seg_ftc.size() - 1] += integral_length;
        
        for (auto loc: seg_ftc)
        {
            h = std::abs(loc - pre);
            if (horizon_project)
            segment_roi = cv::Rect(0, pre, edge_to_project.cols, h);
            else
            {
                segment_roi = cv::Rect(pre, 0, h, edge_to_project.rows);
            }
            segment_rois.push_back(segment_roi);
            pre = loc; 
        }
    }

    return segment_rois;
    
}



std::string filter_string(const std::string filter_string, const std::string ignore_string)
{
    std::string filtered_string = filter_string;
    for (auto c: ignore_string)
    {
        for (auto it = filtered_string.begin(); it != filtered_string.end();)
        {
            if (*it == c)
            filtered_string.erase(it);
            else
            {
                ++it;
            }
            
        }
        
    }
    return filtered_string;
}


//求区域内均值 integral即为积分图
float fastMean(cv::Mat& integral, int x, int y, int window)
{

    int min_y = std::max(0, y - window / 2);
    int max_y = std::min(integral.rows - 1, y + window / 2);
    int min_x = std::max(0, x - window / 2);
    int max_x = std::min(integral.cols - 1, x + window / 2);

    int topright = integral.at<int>(max_y, max_x);
    int botleft = integral.at<int>(min_y, min_x);
    int topleft = integral.at<int>(max_y, min_x);
    int botright = integral.at<int>(min_y, max_x);

    return (float)((topright + botleft - topleft - botright) / (float)((max_y - min_y) *(max_x - min_x)));

}


void binary_sauvola(const cv::Mat& gray_img, cv::Mat& binary_img, int window, float k)
{
    cv::Mat integral_sum, integral_sqsum;
    int nYOffSet = 3;
    int nXOffSet = 3;
    cv::integral(gray_img, integral_sum, integral_sqsum);  //计算积分图像
    std::cout << "gray_img info: " << gray_img.size() << ", " << gray_img.channels();
    int count;
    int xmin, ymin, xmax, ymax;

    int whalf = window >> 1;
    double mean, std, threshold;
    double diagsum, idiagsum, diff, sqdiagsum, sqidiagsum, sqdiff, area;

    for (int i = 0; i < gray_img.cols; ++i){

        for (int j = 0; j < gray_img.rows; ++j){

            xmin = std::max(0, i - whalf);

            ymin = std::max(0, j - whalf);

            xmax = std::min(gray_img.cols - 1, i + whalf);

            ymax = std::min(gray_img.rows - 1, j + whalf);

            area = (xmax - xmin + 1) * (ymax - ymin + 1);

            if (area <= 0){

                binary_img.at<uchar>(j, i) = 255;
                continue;

            }

            if (xmin == 0 && ymin == 0){

                diff = integral_sum.at<uchar>(ymax, xmax);

                sqdiff = integral_sqsum.at<uchar>(ymax, xmax);

            }

            else if (xmin > 0 && ymin == 0){

                diff = integral_sum.at<uchar>(ymax, xmax) - integral_sum.at<uchar>(ymax, xmin - 1);

                sqdiff = integral_sqsum.at<uchar>(ymax, xmax) - integral_sqsum.at<uchar>(ymax, xmin - 1);

            }

            else if (xmin == 0 && ymin > 0){

                diff = integral_sum.at<uchar>(ymax, xmax) - integral_sum.at<uchar>(ymin - 1, xmax);

                sqdiff = integral_sqsum.at<uchar>(ymax, xmax) - integral_sqsum.at<uchar>(ymin - 1, xmax);;

            }

            else{

                diagsum = integral_sum.at<uchar>(ymax, xmax) + integral_sum.at<uchar>(ymin - 1, xmin - 1);

                idiagsum = integral_sum.at<uchar>(ymin - 1, xmax) + integral_sum.at<uchar>(ymax, xmin - 1);

                diff = diagsum - idiagsum;

                sqdiagsum = integral_sqsum.at<uchar>(ymax, xmax) + integral_sqsum.at<uchar>(ymin - 1, xmin - 1);

                sqidiagsum = integral_sqsum.at<uchar>(ymin - 1, xmax) + integral_sqsum.at<uchar>(ymax, xmin - 1);

                sqdiff = sqdiagsum - sqidiagsum;

            }

            mean = diff / area;

            std = sqrt((sqdiff - diff*diff / area) / (area - 1));

            threshold = mean*(1 + k*((std / 128) - 1));

            if ( gray_img.at<uchar>(j, i) < threshold)

                 binary_img.at<uchar>(j, i) = 0;

            else

                 binary_img.at<uchar>(j, i) = 255;

        }

    }
    cv::imwrite("/opt/history/temp_images/sauvola_bin.png", binary_img);
    std::cout << "  white pixel: " << count << std::endl;

    //return resImg;
}


float reblur(const unsigned char *data, int width, int height)
{
    float blur_val = 0.0f;
    float kernel[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
    float *BVer = new float[width * height];//垂直方向低通滤波后的结果
    float *BHor = new float[width * height];//水平方向低通滤波后的结果

    float filter_data = 0.0;
    for (int i = 0; i < height; ++i)//均值滤波
    {
        for (int j = 0; j < width; ++j)
        {
            if (i < 4 || i > height - 5)
            {//处理边界 直接赋值原数据
                BVer[i * width + j] = data[i * width + j];
            }
            else
            {
                filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] + kernel[2] * data[(i - 2) * width + j] +
                    kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i)* width + j] + kernel[5] * data[(i + 1) * width + j] +
                    kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] + kernel[8] * data[(i + 4) * width + j];
                BVer[i * width + j] = filter_data;
            }

            if (j < 4 || j > width - 5)
            {
                BHor[i * width + j] = data[i * width + j];
            }
            else
            {
                filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] + kernel[2] * data[i * width + (j - 2)] +
                    kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + (j + 1)] +
                    kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] + kernel[8] * data[i * width + (j + 4)];
                BHor[i * width + j] = filter_data;
            }

        }
    }

    float D_Fver = 0.0;
    float D_FHor = 0.0;
    float D_BVer = 0.0;
    float D_BHor = 0.0;
    float s_FVer = 0.0;//原始图像数据的垂直差分总和 对应论文中的 s_Fver
    float s_FHor = 0.0;//原始图像数据的水平差分总和 对应论文中的 s_Fhor
    float s_Vver = 0.0;//模糊图像数据的垂直差分总和 s_Vver
    float s_VHor = 0.0;//模糊图像数据的水平差分总和 s_VHor
    for (int i = 1; i < height; ++i)
    {
        for (int j = 1; j < width; ++j)
        {
            D_Fver = std::abs((float)data[i * width + j] - (float)data[(i - 1) * width + j]);
            s_FVer += D_Fver;
            D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
            s_Vver += std::max((float)0.0, D_Fver - D_BVer);

            D_FHor = std::abs((float)data[i * width + j] - (float)data[i * width + (j - 1)]);
            s_FHor += D_FHor;
            D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
            s_VHor += std::max((float)0.0, D_FHor - D_BHor);
        }
    }
    float b_FVer = (s_FVer - s_Vver) / s_FVer;
    float b_FHor = (s_FHor - s_VHor) / s_FHor;
    blur_val = std::max(b_FVer, b_FHor);

    delete[] BVer;
    delete[] BHor;

    return blur_val;
}

float grid_max_reblur(const cv::Mat &img, int rows, int cols)
{
    int row_height = img.rows / rows;
    int col_width = img.cols / cols;
    float blur_val = FLT_MIN;
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            cv::Mat img_roi = img(cv::Rect(x * col_width, y * row_height, col_width, row_height));
            auto this_grad_blur_val = reblur(img_roi.data, img_roi.cols, img_roi.rows);
            if (this_grad_blur_val > blur_val) blur_val = this_grad_blur_val;
        }
    }
    return std::max<float>(blur_val, 0);
}


float clarity_estimate(const cv::Mat& image)
{

    // float blur_val = ReBlur(src_data.data(), src_data.width(), src_data.height());
    float blur_val = grid_max_reblur(image, 2, 2);
    float clarity = 1.0f - blur_val;

    float T1 = 0.0f;
    float T2 = 1.0f;
    if (clarity <= T1)
    {
        clarity = 0.0;
    }
    else if (clarity >= T2)
    {
        clarity = 1.0;
    }
    else
    {
        clarity = (clarity - T1) / (T2 - T1);
    }

    return clarity*1000;
}

