#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

// 压缩包结构：单纯的一串字节，发送给下位机
struct Packet
{
    vector<uint8_t> data;
};

// 处理结果结构体
struct ProcessResult
{
    cv::Mat visualization;         // 原始尺寸的轮廓图
    vector<uint8_t> compressData;  // 仅经过 RLE 压缩的数据
    int ballCount;                 // 检测到的弹丸数
    int contourCount;              // 检测到的轮廓总数
};

class HeroCamCompressor
{
public:
    // ============ 弹丸筛选阈值 ============
    // 注意：代码中目前使用 Canny 边缘检测，若需精准识别，建议后续加入 HSV 掩膜
    const float MIN_BALL_AREA = 3.0f;
    const float MAX_BALL_AREA = 300.0f;
    const float MIN_BALL_CIRCULARITY = 0.7f;
    const float MAX_BALL_ASPECT_RATIO = 1.5f;

    // 目标压缩分辨率
    const Size TARGET_SIZE = Size(160, 112);

    ProcessResult process(Mat &input)
    {
        ProcessResult result;
        if (input.empty()) return result;

        // 1. 预处理
        Mat gray, blurred, edges;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(5, 5), 1.3);
        Canny(blurred, edges, 50, 150);
        
        // 2. 轮廓提取
        vector<vector<Point>> contours;
        findContours(edges.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // 3. 绘制轮廓（原始尺寸可视化）
        Mat visualization = Mat::zeros(input.size(), CV_8UC3);
        drawContours(visualization, contours, -1, Scalar(255, 255, 255), 2);

        Mat kernel1 = getStructuringElement(MORPH_RECT, Size(2,2));
        Mat kernel2 = getStructuringElement(MORPH_RECT, Size(4,4));
        erode(visualization, visualization, kernel1);
        dilate(visualization, visualization, kernel2);

        // 4. 筛选弹丸逻辑
        int validBalls = 0;
        for (const auto &cnt : contours)
        {
            double area = contourArea(cnt);
            if (area < MIN_BALL_AREA || area > MAX_BALL_AREA) continue;

            double perim = arcLength(cnt, true);
            if (perim <= 0) continue;
            double circularity = 4.0 * CV_PI * area / (perim * perim);
            if (circularity < MIN_BALL_CIRCULARITY) continue;

            Rect rect = boundingRect(cnt);
            double aspect = static_cast<double>(rect.width) / rect.height;
            if (aspect < 1.0) aspect = 1.0 / aspect;
            if (aspect > MAX_BALL_ASPECT_RATIO) continue;

            validBalls++;
        }
        
        // 5. 缩放并准备压缩
        Mat resized, binary;
        cv::resize(visualization, resized, TARGET_SIZE, 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);
        cv::threshold(resized, binary, 128, 255, cv::THRESH_BINARY);
        
        // 6. RLE 编码
        result.compressData = compressRLE(binary);
        result.visualization = visualization;
        result.ballCount = validBalls;
        result.contourCount = contours.size();
        
        return result;
    }

private:
    // ============ RLE 压缩核心 ============
    // 存储格式：[长度1, 值1, 长度2, 值2, ...]
    vector<uint8_t> compressRLE(const Mat &img)
    {
        vector<uint8_t> buffer;
        const uchar *ptr = img.data;
        int total = img.rows * img.cols;

        for (int i = 0; i < total; ) {
            uchar val = (ptr[i] > 128) ? 1 : 0; 
            uint8_t count = 1;
            // 统计连续相同像素，且长度不超过 uint8 限制 (255)
            while (i + count < total && 
                  ((ptr[i + count] > 128 ? 1 : 0) == val) && 
                  count < 255) {
                count++;
            }

            buffer.push_back(count);
            buffer.push_back(val);
            i += count;
        }
        return buffer;
    }
};

// ============ 解码函数 ============

// 直接从 RLE 流恢复二值图像
cv::Mat decodeRLE(const std::vector<uint8_t> &rle_data, cv::Size sz)
{
    cv::Mat decoded = cv::Mat::zeros(sz, CV_8UC1);
    if (rle_data.empty()) return decoded;

    uchar *ptr = decoded.data;
    int pixelIdx = 0;
    int totalPixels = sz.width * sz.height;
    
    // 每两个字节为一个单元 (Count, Value)
    for (size_t i = 0; i + 1 < rle_data.size(); i += 2) {
        uint8_t count = rle_data[i];
        uint8_t val = rle_data[i + 1];
        uint8_t pixel_val = (val == 1) ? 255 : 0;
        
        for (int j = 0; j < count && pixelIdx < totalPixels; j++) {
            ptr[pixelIdx++] = pixel_val;
        }
    }
    
    return decoded;
}