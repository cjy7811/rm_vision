#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <queue>
#include <memory>
#include <map>
#include <bitset>

using namespace cv;
using namespace std;

// Huffman 编码相关结构
struct HuffmanNode
{
    int value; // -1 表示内部节点，0-3 表示数据值
    int freq;
    shared_ptr<HuffmanNode> left;
    shared_ptr<HuffmanNode> right;

    HuffmanNode(int v, int f) : value(v), freq(f), left(nullptr), right(nullptr) {}
};

struct HuffmanCompare
{
    bool operator()(shared_ptr<HuffmanNode> a, shared_ptr<HuffmanNode> b) const
    {
        return a->freq > b->freq;
    }
};

// 压缩包结构：单纯的一串字节，发给电控去拆包
struct Packet
{
    vector<uint8_t> data;
};

class HeroCamCompressor
{
public:
    // 针对视频调的阈值 (HSV空间)
    // 绿色弹丸
    Scalar lowGreen = Scalar(35, 40, 40);
    Scalar highGreen = Scalar(90, 255, 255);

    // 目标压缩后分辨率（略微降低高度以满足带宽要求）
    const Size TARGET_SIZE = Size(160, 112);

    vector<uint8_t> process(Mat &input)
    {
        if (input.empty())
        {
            cerr << "HeroCamCompressor::process: input empty" << endl;
            return {};
        }
        Mat hsv, combinedMask, cleanMask;
        Mat resizedGray, finalMap;

        // 1. 颜色提取（放宽绿色范围，夜景更稳）
        cvtColor(input, hsv, COLOR_BGR2HSV);

        if (hsv.empty())
            return {};

        Mat maskGreen;
        inRange(hsv,
                Scalar(35, 80, 80),
                Scalar(80, 255, 255),
                maskGreen);

        Mat maskBright;
        // maskBright = Mat::zeros(hsv.size(), CV_8UC1);

        // vector<Mat> hsvSplit;
        // split(hsv, hsvSplit);

        // if (hsvSplit.size() == 3)
        // {
        //     Mat V = hsvSplit[2];
        //     threshold(V, maskBright, 230, 255, THRESH_BINARY);
        // }

        // bitwise_or(maskGreen, maskBright, combinedMask);
        combinedMask = maskGreen;

        // 1.5 轻微开运算去孤立噪点（避免芝麻点）
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(combinedMask, combinedMask, MORPH_OPEN, kernel);

        // 2. 找轮廓
        vector<vector<Point>> contours;
        findContours(combinedMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        cleanMask = Mat::zeros(combinedMask.size(), CV_8UC1);

        for (const auto &cnt : contours)
        {
            double area = contourArea(cnt);
            Rect rect = boundingRect(cnt);

            // 关键修改：
            // 不再用单一 >10，而是面积区间 + 尺寸判断
            if (area > 3 && area < 800)
            {
                if (rect.width >= 2 && rect.height >= 2)
                {
                    drawContours(cleanMask, vector<vector<Point>>{cnt}, -1, 255, FILLED);
                }
            }
        }

        // 3. 膨胀（稍微小一点，避免吃掉周围）
        Mat dilatedMask;
        Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
        dilate(cleanMask, dilatedMask, element);

        // 4. 灰度图
        Mat grayFull;
        cvtColor(input, grayFull, COLOR_BGR2GRAY);

        Mat maskCoreSmall, maskSurroundSmall;
        resize(cleanMask, maskCoreSmall, TARGET_SIZE, 0, 0, INTER_NEAREST);
        resize(dilatedMask, maskSurroundSmall, TARGET_SIZE, 0, 0, INTER_NEAREST);

        // 5. 四值化
        finalMap = Mat::zeros(TARGET_SIZE, CV_8UC1);

        for (int i = 0; i < finalMap.rows; i++)
        {
            for (int j = 0; j < finalMap.cols; j++)
            {
                int srcY = i * grayFull.rows / TARGET_SIZE.height;
                int srcX = j * grayFull.cols / TARGET_SIZE.width;

                if (maskCoreSmall.at<uchar>(i, j) > 0)
                {
                    finalMap.at<uchar>(i, j) = 3;
                }
                else if (maskSurroundSmall.at<uchar>(i, j) > 0)
                {
                    uchar g = grayFull.at<uchar>(srcY, srcX);
                    finalMap.at<uchar>(i, j) = (g > 60) ? 2 : 1;
                }
                else
                {
                    uchar g = grayFull.at<uchar>(srcY, srcX);

                    // 背景别压太狠，夜景容易把弹丸吃掉
                    if (g < 25)
                        finalMap.at<uchar>(i, j) = 0;
                    else if (g < 70)
                        finalMap.at<uchar>(i, j) = 1;
                    else
                        finalMap.at<uchar>(i, j) = 2;
                }
            }
        }

        // 降噪以减少孤立噪点，提升 RLE 压缩效率
        medianBlur(finalMap, finalMap, 3);

        vector<uint8_t> rleData = compressRLE(finalMap);
        return compressHuffman(rleData);
    }

private:
    // ============ Huffman 编码实现 ============
    vector<uint8_t> compressHuffman(const vector<uint8_t> &rleData)
    {
        if (rleData.empty())
            return {};

        // 如果 RLE 数据本身就很小，跳过 Huffman，直接发送原始 RLE 数据
        // 格式：0xFF + 原始长度(4字节) + 原始数据
        const size_t RAW_THRESHOLD = 200;
        if (rleData.size() <= RAW_THRESHOLD)
        {
            vector<uint8_t> raw;
            raw.push_back(0xFF);
            uint32_t len = (uint32_t)rleData.size();
            raw.push_back((len >> 24) & 0xFF);
            raw.push_back((len >> 16) & 0xFF);
            raw.push_back((len >> 8) & 0xFF);
            raw.push_back(len & 0xFF);
            raw.insert(raw.end(), rleData.begin(), rleData.end());
            return raw;
        }

        // 1. 统计频率
        map<uint8_t, int> freqMap;
        for (uint8_t byte : rleData)
        {
            freqMap[byte]++;
        }

        // 2. 构建 Huffman 树
        priority_queue<shared_ptr<HuffmanNode>, vector<shared_ptr<HuffmanNode>>, HuffmanCompare> pq;
        for (auto &p : freqMap)
        {
            pq.push(make_shared<HuffmanNode>(p.first, p.second));
        }

        // 特殊情况：只有一个符号
        if (pq.size() == 1)
        {
            auto single = pq.top();
            auto node = make_shared<HuffmanNode>(-1, single->freq);
            node->left = single;
            pq.pop();
            pq.push(node);
        }

        while (pq.size() > 1)
        {
            auto left = pq.top();
            pq.pop();
            auto right = pq.top();
            pq.pop();
            auto parent = make_shared<HuffmanNode>(-1, left->freq + right->freq);
            parent->left = left;
            parent->right = right;
            pq.push(parent);
        }

        auto root = pq.top();

        // 3. 生成编码表
        map<uint8_t, string> codeTable;
        buildCodeTable(root, "", codeTable);

        // 4. 编码数据
        string bitString;
        for (uint8_t byte : rleData)
        {
            bitString += codeTable[byte];
        }

        // 5. 序列化：头部 + 频率表 + 数据
        vector<uint8_t> result;

        // 头部：频率表大小（1字节）
        result.push_back((uint8_t)freqMap.size());

        // 频率表：每个符号及其频率
        for (auto &p : freqMap)
        {
            result.push_back(p.first); // 符号
            uint32_t freq = p.second;
            result.push_back((freq >> 24) & 0xFF);
            result.push_back((freq >> 16) & 0xFF);
            result.push_back((freq >> 8) & 0xFF);
            result.push_back(freq & 0xFF);
        }

        // 原始RLE数据长度（4字节）
        uint32_t dataLen = rleData.size();
        result.push_back((dataLen >> 24) & 0xFF);
        result.push_back((dataLen >> 16) & 0xFF);
        result.push_back((dataLen >> 8) & 0xFF);
        result.push_back(dataLen & 0xFF);

        // 压缩后数据长度（4字节）
        uint32_t compLen = (bitString.length() + 7) / 8;
        result.push_back((compLen >> 24) & 0xFF);
        result.push_back((compLen >> 16) & 0xFF);
        result.push_back((compLen >> 8) & 0xFF);
        result.push_back(compLen & 0xFF);

        // 位数长度（4字节）
        uint32_t bitLen = bitString.length();
        result.push_back((bitLen >> 24) & 0xFF);
        result.push_back((bitLen >> 16) & 0xFF);
        result.push_back((bitLen >> 8) & 0xFF);
        result.push_back(bitLen & 0xFF);

        // 位数据：8位打包成1字节
        for (size_t i = 0; i < bitString.length(); i += 8)
        {
            uint8_t byte = 0;
            for (int j = 0; j < 8 && i + j < bitString.length(); j++)
            {
                byte = (byte << 1) | (bitString[i + j] - '0');
            }
            // 最后一个字节可能不满8位，左移补齐
            if (i + 8 >= bitString.length())
            {
                byte <<= (8 - (bitString.length() % 8));
            }
            result.push_back(byte);
        }

        return result;
    }

    void buildCodeTable(shared_ptr<HuffmanNode> node, string code, map<uint8_t, string> &table)
    {
        if (!node)
            return;
        if (node->value != -1)
        {
            table[node->value] = code.empty() ? "0" : code;
            return;
        }
        buildCodeTable(node->left, code + "0", table);
        buildCodeTable(node->right, code + "1", table);
    }

    // RLE 算法保持不变，这在黑背景下效率无敌
    vector<uint8_t> compressRLE(const Mat &img)
    {
        vector<uint8_t> buffer;
        if (img.empty())
            return buffer;

        Mat flat = img.reshape(1, 1);
        uchar *p = flat.ptr<uchar>(0);
        int totalPixels = flat.cols;

        int count = 1;
        uchar currentVal = p[0];

        for (int i = 1; i < totalPixels; i++)
        {
            uchar nextVal = p[i];

            if (nextVal == currentVal && count < 63)
            {
                count++;
            }
            else
            {
                // 构造字节：count (6bit) | value (2bit)
                uint8_t byte = (count << 2) | (currentVal & 0x03);
                buffer.push_back(byte);
                currentVal = nextVal;
                count = 1;
            }
        }
        buffer.push_back((count << 2) | (currentVal & 0x03));
        return buffer;
    }
};

// 解码还原（模拟客户端行为）
cv::Mat decodeRLE(const std::vector<uint8_t> &stream, cv::Size sz)
{
    // 1. 先建一张全黑的空图
    cv::Mat decoded = cv::Mat::zeros(sz, CV_8UC1);
    uchar *ptr = decoded.data;
    int pixelIdx = 0;
    int totalPixels = sz.width * sz.height;

    // 2. 遍历压缩包里的每一个字节
    for (uint8_t byte : stream)
    {
        int count = byte >> 2; // 右移两位，拿到前6位的“长度”
        int val = byte & 0x03; // 与运算，拿到最后2位的“颜色等级”

        // 3. 把 0,1,2,3 还原成肉眼可见的灰度值
        uchar grayVal;
        if (val == 3)
            grayVal = 255; // 弹丸/英雄，最亮
        else if (val == 2)
            grayVal = 200; // 提亮背景高光
        else if (val == 1)
            grayVal = 100; // 轮廓
        else
            grayVal = 0; // 背景，全黑

        // 4. 填充像素
        for (int i = 0; i < count && pixelIdx < totalPixels; ++i)
        {
            ptr[pixelIdx++] = grayVal;
        }
    }
    return decoded;
}

// ============ Huffman解码 ============
vector<uint8_t> decompressHuffman(const vector<uint8_t> &compressed)
{
    if (compressed.empty())
        return {};

    size_t pos = 0;

    // 支持原始 RLE 直发格式（0xFF 标记）
    if (compressed[0] == 0xFF)
    {
        if (compressed.size() < 5)
            return {};
        size_t p = 1;
        uint32_t len = 0;
        len |= ((uint32_t)compressed[p++] << 24);
        len |= ((uint32_t)compressed[p++] << 16);
        len |= ((uint32_t)compressed[p++] << 8);
        len |= (uint32_t)compressed[p++];
        if (compressed.size() < 5 + len)
            return {};
        vector<uint8_t> out;
        out.insert(out.end(), compressed.begin() + 5, compressed.begin() + 5 + len);
        return out;
    }
    // 1. 读取频率表大小
    uint8_t freqTableSize = compressed[pos++];

    // 2. 重建频率表
    map<uint8_t, int> freqMap;
    for (int i = 0; i < freqTableSize; i++)
    {
        uint8_t symbol = compressed[pos++];
        uint32_t freq = 0;
        freq |= ((uint32_t)compressed[pos++] << 24);
        freq |= ((uint32_t)compressed[pos++] << 16);
        freq |= ((uint32_t)compressed[pos++] << 8);
        freq |= (uint32_t)compressed[pos++];
        freqMap[symbol] = freq;
    }

    // 3. 读取长度信息
    uint32_t rleDataLen = 0;
    rleDataLen |= ((uint32_t)compressed[pos++] << 24);
    rleDataLen |= ((uint32_t)compressed[pos++] << 16);
    rleDataLen |= ((uint32_t)compressed[pos++] << 8);
    rleDataLen |= (uint32_t)compressed[pos++];

    uint32_t compLen = 0;
    compLen |= ((uint32_t)compressed[pos++] << 24);
    compLen |= ((uint32_t)compressed[pos++] << 16);
    compLen |= ((uint32_t)compressed[pos++] << 8);
    compLen |= (uint32_t)compressed[pos++];

    uint32_t bitLen = 0;
    bitLen |= ((uint32_t)compressed[pos++] << 24);
    bitLen |= ((uint32_t)compressed[pos++] << 16);
    bitLen |= ((uint32_t)compressed[pos++] << 8);
    bitLen |= (uint32_t)compressed[pos++];

    // 4. 重建Huffman树
    priority_queue<shared_ptr<HuffmanNode>, vector<shared_ptr<HuffmanNode>>, HuffmanCompare> pq;
    for (auto &p : freqMap)
    {
        pq.push(make_shared<HuffmanNode>(p.first, p.second));
    }

    if (pq.size() == 1)
    {
        auto single = pq.top();
        auto node = make_shared<HuffmanNode>(-1, single->freq);
        node->left = single;
        pq.pop();
        pq.push(node);
    }

    while (pq.size() > 1)
    {
        auto left = pq.top();
        pq.pop();
        auto right = pq.top();
        pq.pop();
        auto parent = make_shared<HuffmanNode>(-1, left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    auto root = pq.top();

    // 5. 解压缩数据
    string bitString;
    for (uint32_t i = 0; i < compLen && pos < compressed.size(); i++)
    {
        uint8_t byte = compressed[pos++];
        for (int j = 7; j >= 0; j--)
        {
            bitString += ((byte >> j) & 1) ? '1' : '0';
            if ((int)bitString.length() == bitLen)
                break;
        }
    }
    bitString = bitString.substr(0, bitLen);

    // 6. 解码
    vector<uint8_t> rleData;
    auto current = root;
    for (char bit : bitString)
    {
        current = (bit == '0') ? current->left : current->right;
        if (current->value != -1)
        {
            rleData.push_back(current->value);
            current = root;
            if ((int)rleData.size() == rleDataLen)
                break;
        }
    }

    return rleData;
}

// ============ 完整解码流程 ============
cv::Mat decodeCompressed(const vector<uint8_t> &compressed, cv::Size sz)
{
    // 先Huffman解码得到RLE数据，再RLE解码
    vector<uint8_t> rleData = decompressHuffman(compressed);
    return decodeRLE(rleData, sz);
}