#ifndef HEADER_H
#define HEADER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <cstring>

// ============ 常量定义 ============
constexpr int TOTAL_PACKET_BYTE = 300;
constexpr int RLE_DATA_MAX_BYTE = 275;
constexpr int HEADER_BYTE = 16;
constexpr int RESERVED_BYTE = 9;
static_assert(HEADER_BYTE + RLE_DATA_MAX_BYTE + RESERVED_BYTE == TOTAL_PACKET_BYTE,
              "300字节硬约束校验失败");

const cv::Size TARGET_SIZE(120, 80);

// ============ 弹丸识别参数 ============
constexpr float MIN_BALL_AREA = 3.0f;
constexpr float MAX_BALL_AREA = 2000.0f;
constexpr float MIN_BALL_CIRCULARITY = 0.85f;
constexpr float MAX_BALL_ASPECT_RATIO = 1.3f;
const cv::Scalar BALL_HSV_LOW(40, 10, 150);
const cv::Scalar BALL_HSV_HIGH(95, 255, 255);

// ============ 数据结构 ============
#pragma pack(1)
struct BallInfo {
    uint8_t x;  // 小分辨率X坐标
    uint8_t y;  // 小分辨率Y坐标
    uint8_t r;  // 小分辨率半径
};

struct MqttPacket {
    uint8_t frame_seq;
    uint8_t config;
    uint8_t width;
    uint8_t height;
    BallInfo balls[4];                // 最多4个弹丸
    uint8_t rle_data[RLE_DATA_MAX_BYTE];
    uint8_t reserved[RESERVED_BYTE];
};
#pragma pack()
static_assert(sizeof(MqttPacket) == TOTAL_PACKET_BYTE, "数据包必须严格300字节");

struct ProcessResult {
    cv::Mat finalBinary;
    cv::Mat originalMarked;
    MqttPacket packet;
    int rle_used_byte;
    int ballCount;
    std::vector<cv::Point2f> ballCenters;
    std::vector<float> ballRadii;
};

// ============ 核心压缩器类声明 ============
class HeroCamCompressor {
public:
    ProcessResult process(cv::Mat& input);

private:
    int compressRLE(const cv::Mat& img, uint8_t* out_buf, int max_len);
};

// ============ 辅助函数声明 ============
cv::Mat decodeRLE(const uint8_t* rle_data, int rle_len, cv::Size sz);
bool createDir(const std::string& path);

#endif // HEADER_H