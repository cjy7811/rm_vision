#include "thread.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>  // for max_element
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ============ 全局变量定义 ============
RingBuffer frame_queue(100);
std::mutex camera_mutex;
std::condition_variable frame_available;
std::condition_variable queue_not_full;
std::atomic<bool> running{true};
int frame_skip = 1;

// ============ HeroCamCompressor 成员函数实现 ============
ProcessResult HeroCamCompressor::process(Mat& input) {
    ProcessResult result;
    if (input.empty()) return result;
    int origW = input.cols;
    int origH = input.rows;

    // 1. Canny赛场轮廓提取
    Mat gray, blurred, edges;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 1.3);
    Canny(blurred, edges, 50, 150);
    
    vector<vector<Point>> contours;
    findContours(edges.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat visualization = Mat::zeros(input.size(), CV_8UC1);
    drawContours(visualization, contours, -1, Scalar(255), 2);

    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(2,2));
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(4,4));
    erode(visualization, visualization, kernel1);
    dilate(visualization, visualization, kernel2);

    // 2. HSV绿色弹丸提取
    Mat hsv, greenMask;
    cvtColor(input, hsv, COLOR_BGR2HSV);
    inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH, greenMask);
    morphologyEx(greenMask, greenMask, MORPH_CLOSE, kernel1);
    dilate(greenMask, greenMask, kernel2);

    vector<vector<Point>> ballContours;
    findContours(greenMask.clone(), ballContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 按面积从大到小排序
    sort(ballContours.begin(), ballContours.end(),
         [](const vector<Point>& a, const vector<Point>& b) {
             return contourArea(a) > contourArea(b);
         });

    int validBalls = 0;
    Mat originalMarked;
    cvtColor(visualization, originalMarked, COLOR_GRAY2BGR);

    // 初始化数据包
    MqttPacket pkt;
    memset(&pkt, 0, sizeof(MqttPacket));

    for (const auto &cnt : ballContours) {
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

        // 记录弹丸信息
        Point2f center;
        float radius;
        minEnclosingCircle(cnt, center, radius);
        result.ballCenters.push_back(center);
        result.ballRadii.push_back(radius);

        // 原始画面标记
        circle(originalMarked, center, (int)radius, Scalar(255, 255, 255), -1);
        circle(originalMarked, center, (int)radius + 3, Scalar(0, 255, 0), 3);

        if (validBalls < 4) {
            pkt.balls[validBalls].x = (uint8_t)cvRound(center.x * TARGET_SIZE.width / origW);
            pkt.balls[validBalls].y = (uint8_t)cvRound(center.y * TARGET_SIZE.height / origH);
            pkt.balls[validBalls].r = (uint8_t)cvRound(radius * TARGET_SIZE.width / origW);
            validBalls++;
        }
    }

    // 合并弹丸像素
    bitwise_or(visualization, greenMask, visualization);

    // 缩放+RLE压缩
    Mat resized, binary;
    resize(visualization, resized, TARGET_SIZE, 0, 0, INTER_LINEAR);
    threshold(resized, binary, 128, 255, THRESH_BINARY);
    
    pkt.config = 0x01;
    pkt.width = TARGET_SIZE.width;
    pkt.height = TARGET_SIZE.height;
    int rle_len = compressRLE(binary, pkt.rle_data, RLE_DATA_MAX_BYTE);
    if (rle_len >= RLE_DATA_MAX_BYTE) pkt.config |= 0x02;

    result.finalBinary = binary;
    result.originalMarked = originalMarked;
    result.packet = pkt;
    result.rle_used_byte = rle_len;
    result.ballCount = validBalls;
    
    return result;
}

int HeroCamCompressor::compressRLE(const Mat& img, uint8_t* out_buf, int max_len) {
    int buf_idx = 0;
    const uchar *ptr = img.data;
    int total = img.rows * img.cols;
    for (int i = 0; i < total && buf_idx + 1 < max_len; ) {
        uchar val = (ptr[i] > 128) ? 1 : 0; 
        uint8_t count = 1;
        while (i + count < total &&
               ((ptr[i + count] > 128 ? 1 : 0) == val) &&
               count < 255 &&
               buf_idx + 1 < max_len)
            count++;
        out_buf[buf_idx++] = count;
        out_buf[buf_idx++] = val;
        i += count;
    }
    return buf_idx;
}

// ============ 辅助函数实现 ============
Mat decodeRLE(const uint8_t* rle_data, int rle_len, Size sz) {
    Mat decoded = Mat::zeros(sz, CV_8UC1);
    if (rle_len <= 0) return decoded;
    uchar *ptr = decoded.data;
    int pixelIdx = 0;
    int totalPixels = sz.width * sz.height;
    for (int i = 0; i + 1 < rle_len; i += 2) {
        uint8_t count = rle_data[i];
        uint8_t val = rle_data[i + 1];
        uint8_t pixel_val = (val == 1) ? 255 : 0;
        for (int j = 0; j < count && pixelIdx < totalPixels; j++)
            ptr[pixelIdx++] = pixel_val;
    }
    return decoded;
}

bool createDir(const string& path) {
    return system(("mkdir -p " + path).c_str()) == 0;
}

// ============ RingBuffer 成员函数实现 ============
RingBuffer::RingBuffer(int capacity)
    : capacity_(capacity), size_(0), head_(0), tail_(0), buffer_(capacity) {}

bool RingBuffer::push(Mat&& frame) {
    if (size_ >= capacity_) return false;
    buffer_[tail_] = std::move(frame);
    tail_ = (tail_ + 1) % capacity_;
    size_++;
    return true;
}

bool RingBuffer::pop(Mat& frame) {
    if (size_ == 0) return false;
    frame = std::move(buffer_[head_]);
    head_ = (head_ + 1) % capacity_;
    size_--;
    return true;
}

int RingBuffer::size() const { return size_; }
int RingBuffer::capacity() const { return capacity_; }
bool RingBuffer::empty() const { return size_ == 0; }
bool RingBuffer::full() const { return size_ >= capacity_; }

// ============ 单线程模式实现 ============
void run_single_thread_mode(const string& source) {
    VideoCapture cap(source);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file: " << source << endl;
        return;
    }
    
    double video_fps = cap.get(CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 30.0;
    cout << "Video FPS: " << fixed << setprecision(2) << video_fps << endl;
    
    int origWidth = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int origHeight = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    
    long custom_spf = (long)(1000.0 / video_fps);
    
    HeroCamCompressor compressor;
    Mat frame;
    ProcessResult result;
    
    const string OUTPUT_VIDEO_PATH = "output_video.avi";
    const string OUTPUT_FRAMES_DIR = "output_frames/";
    
    if (!createDir(OUTPUT_FRAMES_DIR)) {
        cerr << "[错误] 无法创建目录: " << OUTPUT_FRAMES_DIR << endl;
    }
    
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer(OUTPUT_VIDEO_PATH, fourcc, video_fps,
                       Size(origWidth * 2, origHeight), true);
    if (!writer.isOpened()) {
        cerr << "[错误] 无法创建输出视频文件: " << OUTPUT_VIDEO_PATH << endl;
    }
    
    const string window_name = "Operator View (Single-thread)";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, 1280, 480);
    
    Mat displayImg;
    displayImg.create(origHeight, origWidth * 2, CV_8UC3);
    
    int total_frames = 0;
    vector<int> rle_used_sizes;   // 存储RLE实际使用字节数
    vector<int> raw_binary_sizes;  // [新增] 存储压缩前binary每帧大小
    vector<long> frame_times;
    auto last_log_time = high_resolution_clock::now();
    
    uint8_t frame_seq = 0;
    
    cout << "单包固定大小: " << sizeof(MqttPacket) << " 字节 (其中RLE数据区最大 " 
         << RLE_DATA_MAX_BYTE << " 字节)" << endl;
    
    while (cap.read(frame)) {
        if (frame.empty()) continue;
        frame_seq++;
        
        auto start = high_resolution_clock::now();
        
        try {
            result = compressor.process(frame);
            result.packet.frame_seq = frame_seq;
            
            // [新增] 计算压缩前binary大小 (TARGET_SIZE为固定尺寸，每像素1字节)
            int raw_size = TARGET_SIZE.width * TARGET_SIZE.height;
            raw_binary_sizes.push_back(raw_size);
            
            Mat decoded_small = decodeRLE(result.packet.rle_data,
                                          RLE_DATA_MAX_BYTE, TARGET_SIZE);
            Mat decoded_full;
            resize(decoded_small, decoded_full,
                   Size(origWidth, origHeight), 0, 0, INTER_NEAREST);
            Mat decoded_display;
            cvtColor(decoded_full, decoded_display, COLOR_GRAY2BGR);
            
            for (int i = 0; i < 4; i++) {
                if (result.packet.balls[i].x != 0 || result.packet.balls[i].y != 0) {
                    int real_radius = cvRound(result.packet.balls[i].r *
                                              origWidth / TARGET_SIZE.width);
                    Point center(
                        cvRound(result.packet.balls[i].x * origWidth / TARGET_SIZE.width),
                        cvRound(result.packet.balls[i].y * origHeight / TARGET_SIZE.height)
                    );
                    circle(decoded_display, center, real_radius,
                           Scalar(255, 255, 255), -1);
                    circle(decoded_display, center, real_radius + 3,
                           Scalar(0, 255, 0), 3);
                }
            }
            
            result.originalMarked.copyTo(displayImg(Rect(0, 0, origWidth, origHeight)));
            decoded_display.copyTo(displayImg(Rect(origWidth, 0, origWidth, origHeight)));
            
            putText(displayImg, "Original", Point(20, 40),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            putText(displayImg, "Decoded", Point(origWidth + 20, 40),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
            
            imshow(window_name, displayImg);
            
            if (writer.isOpened()) {
                writer.write(displayImg);
                char frame_path[256];
                sprintf(frame_path, "%s/frame_%06d.png",
                        OUTPUT_FRAMES_DIR.c_str(), total_frames + 1);
                imwrite(frame_path, displayImg);
            }
            
            int key = waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
            
            total_frames++;
            rle_used_sizes.push_back(result.rle_used_byte);
            frame_times.push_back(duration_cast<milliseconds>(
                high_resolution_clock::now() - start).count());
            
            auto now = high_resolution_clock::now();
            if (duration_cast<seconds>(now - last_log_time).count() >= 5) {
                double fps = 1000.0 / custom_spf;
                int max_rle_used = *max_element(rle_used_sizes.begin(), rle_used_sizes.end());
                // [新增] 计算压缩前binary的最大值（实际上恒定）
                int max_raw_binary = *max_element(raw_binary_sizes.begin(), raw_binary_sizes.end());
                long avg_time = accumulate(frame_times.begin(),
                                           frame_times.end(), 0L) /
                               (frame_times.empty() ? 1 : frame_times.size());
                
                // 检查最大值是否接近或超过RLE数据区上限
                if (max_rle_used >= RLE_DATA_MAX_BYTE) {
                    cout << "[警告] RLE数据最大值达到或超过上限 (" << max_rle_used
                         << "/" << RLE_DATA_MAX_BYTE << " 字节)" << endl;
                }
                
                cout << "\n[Frame " << total_frames << "] ===== STATISTICS =====" << endl;
                cout << "FPS: " << fixed << setprecision(1) << fps << " fps" << endl;
                cout << "Packet Size (fixed): " << sizeof(MqttPacket) << " bytes" << endl;
                // [新增] 输出压缩前binary大小
                cout << "Raw Binary Size: " << TARGET_SIZE.width << " x " << TARGET_SIZE.height 
                     << " = " << max_raw_binary << " bytes (fixed)" << endl;
                cout << "RLE Data Max Used: " << max_rle_used << " / " 
                     << RLE_DATA_MAX_BYTE << " bytes" << endl;
                cout << "Avg Process Time: " << avg_time << " ms" << endl;
                cout << "========================" << endl;
                
                last_log_time = now;
                frame_times.clear();
                rle_used_sizes.clear();
                raw_binary_sizes.clear();  // [新增] 清空以便下一周期
            }
            
        } catch (const exception& e) {
            cerr << "Error processing frame: " << e.what() << endl;
        }
        
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<milliseconds>(end - start);
        long sleep_time = custom_spf - dur.count();
        if (sleep_time > 2) {
            this_thread::sleep_for(milliseconds(sleep_time));
        }
    }
    
    cap.release();
    writer.release();
    destroyAllWindows();
    
    cout << "Single-thread mode completed. Total frames: " << total_frames << endl;
    cout << "Output video saved to: " << OUTPUT_VIDEO_PATH << endl;
    cout << "Frames saved to: " << OUTPUT_FRAMES_DIR << endl;
}

// ============ 摄像头线程函数实现 ============
int camera_thread_func(const string& source) {
    VideoCapture cap;
    double source_fps = 30.0;
    
    if (source == "0") {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera" << endl;
            return -1;
        }
        cap.set(CAP_PROP_BUFFERSIZE, 1);
        source_fps = cap.get(CAP_PROP_FPS);
    } else {
        cap.open(source);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video file: " << source << endl;
            return -1;
        }
        source_fps = cap.get(CAP_PROP_FPS);
        cout << "Video FPS: " << source_fps << endl;
    }

    Mat frame;
    int frame_count = 0;
    
    while (running) {
        if (!cap.read(frame)) break;
        if (frame.empty()) continue;
        
        if (frame_count % frame_skip != 0) {
            frame_count++;
            continue;
        }
        
        {
            unique_lock<mutex> lock(camera_mutex);
            queue_not_full.wait(lock, []{ return !running || !frame_queue.full(); });
            if (!running) break;
            if (!frame_queue.push(std::move(frame))) {
                cerr << "Failed to push frame to queue" << endl;
            }
            lock.unlock();
            frame_available.notify_one();
        }
        
        frame_count++;
    }
    
    {
        lock_guard<mutex> lock(camera_mutex);
        running = false;
        frame_available.notify_all();
        queue_not_full.notify_all();
    }
    
    cap.release();
    return 0;
}

// ============ 主函数 ============
int main() {
    cout << "=== Image Source Selection ===" << endl;
    cout << "1. Camera (Multi-thread mode) (press 1)" << endl;
    cout << "2. Video File (Single-thread mode) (press 2)" << endl;
    cout << "Please select (1 or 2): ";
    
    char choice;
    cin >> choice;
    
    string source;
    bool use_camera = false;
    
    if (choice == '1') {
        source = "0";
        use_camera = true;
        cout << "Using camera as source (Multi-thread mode)..." << endl;
    } else {
        source = "../vid/test_video1.mp4";  // 默认视频文件路径
        cout << "Using video file (Single-thread mode): " << source << endl;
    }
    
    cout << endl;
    
    if (use_camera) {
        // 多线程模式
        cout << "[Multi-thread mode] Starting..." << endl;
        running = true;
        
        thread camera_thread(camera_thread_func, source);
        
        HeroCamCompressor compressor;
        ProcessResult result;
        long custom_spf = 33;  // 约30fps
        
        const string window_name = "Operator View (Multi-thread)";
        namedWindow(window_name, WINDOW_NORMAL);
        resizeWindow(window_name, 1280, 480);
        
        int origWidth = 640, origHeight = 480;  // 默认，随后从第一帧更新
        Mat displayImg;
        
        // [新增] 多线程模式统计变量
        struct {
            int total_frames = 0;
            vector<long> frame_times;
            vector<int> compressed_sizes;   // RLE压缩后大小
            vector<int> raw_binary_sizes;    // 压缩前binary大小
        } stats;
        
        auto last_log_time = high_resolution_clock::now();
        
        cout << "单包固定大小: " << sizeof(MqttPacket) << " 字节 (其中RLE数据区最大 " 
             << RLE_DATA_MAX_BYTE << " 字节)" << endl;
        
        while(running) {
            auto start = high_resolution_clock::now();
            
            Mat frame;
            {
                unique_lock<mutex> lock(camera_mutex);
                frame_available.wait_for(lock, milliseconds(50), []{
                    return !running || !frame_queue.empty();
                });
                if (!running && frame_queue.empty()) break;
                if (frame_queue.empty()) continue;
                if (!frame_queue.pop(frame)) continue;
                lock.unlock();
                queue_not_full.notify_one();
            }
            
            if (frame.empty()) continue;
            
            // 更新分辨率（基于第一帧）
            if (origWidth == 640 && origHeight == 480) {
                origWidth = frame.cols;
                origHeight = frame.rows;
            }
            
            try {
                result = compressor.process(frame);
                
                // [新增] 记录压缩前binary大小
                int raw_size = TARGET_SIZE.width * TARGET_SIZE.height;
                stats.raw_binary_sizes.push_back(raw_size);
                
                Mat decoded_small = decodeRLE(result.packet.rle_data,
                                              RLE_DATA_MAX_BYTE, TARGET_SIZE);
                Mat decoded_full;
                resize(decoded_small, decoded_full,
                       Size(origWidth, origHeight), 0, 0, INTER_NEAREST);
                Mat decoded_display;
                cvtColor(decoded_full, decoded_display, COLOR_GRAY2BGR);
                
                for (int i = 0; i < 4; i++) {
                    if (result.packet.balls[i].x != 0 || result.packet.balls[i].y != 0) {
                        int real_radius = cvRound(result.packet.balls[i].r *
                                                  origWidth / TARGET_SIZE.width);
                        Point center(
                            cvRound(result.packet.balls[i].x * origWidth / TARGET_SIZE.width),
                            cvRound(result.packet.balls[i].y * origHeight / TARGET_SIZE.height)
                        );
                        circle(decoded_display, center, real_radius,
                               Scalar(255, 255, 255), -1);
                        circle(decoded_display, center, real_radius + 3,
                               Scalar(0, 255, 0), 3);
                    }
                }
                
                displayImg.create(origHeight, origWidth * 2, CV_8UC3);
                result.originalMarked.copyTo(displayImg(Rect(0, 0, origWidth, origHeight)));
                decoded_display.copyTo(displayImg(Rect(origWidth, 0, origWidth, origHeight)));
                
                putText(displayImg, "Original", Point(20, 40),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
                putText(displayImg, "Decoded", Point(origWidth + 20, 40),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
                
                imshow(window_name, displayImg);
                
                int key = waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    running = false;
                    break;
                }
                
                stats.frame_times.push_back(duration_cast<milliseconds>(
                    high_resolution_clock::now() - start).count());
                stats.compressed_sizes.push_back(result.rle_used_byte);
                stats.total_frames++;
                
                auto now = high_resolution_clock::now();
                if (duration_cast<seconds>(now - last_log_time).count() >= 5) {
                    double fps = 1000.0 / 33.0;  // 假设30FPS
                    int max_rle_used = *max_element(stats.compressed_sizes.begin(),
                                                     stats.compressed_sizes.end());
                    int max_raw_binary = *max_element(stats.raw_binary_sizes.begin(),
                                                       stats.raw_binary_sizes.end());
                    long avg_time = accumulate(stats.frame_times.begin(),
                                               stats.frame_times.end(), 0L) /
                                   (stats.frame_times.empty() ? 1 :
                                    stats.frame_times.size());
                    
                    if (max_rle_used >= RLE_DATA_MAX_BYTE) {
                        cout << "[警告] RLE数据最大值达到或超过上限 (" << max_rle_used
                             << "/" << RLE_DATA_MAX_BYTE << " 字节)" << endl;
                    }
                    
                    cout << "\n[Frame " << stats.total_frames << "] ===== STATISTICS =====" << endl;
                    cout << "FPS: " << fixed << setprecision(1) << fps << " fps" << endl;
                    cout << "Packet Size (fixed): " << sizeof(MqttPacket) << " bytes" << endl;
                    // [新增] 输出压缩前binary大小
                    cout << "Raw Binary Size: " << TARGET_SIZE.width << " x " << TARGET_SIZE.height 
                         << " = " << max_raw_binary << " bytes (fixed)" << endl;
                    cout << "RLE Data Max Used: " << max_rle_used << " / " 
                         << RLE_DATA_MAX_BYTE << " bytes" << endl;
                    cout << "Avg Process Time: " << avg_time << " ms" << endl;
                    cout << "========================" << endl;
                    
                    last_log_time = now;
                    stats.frame_times.clear();
                    stats.compressed_sizes.clear();
                    stats.raw_binary_sizes.clear();  // [新增] 清空
                }
                
            } catch (const exception& e) {
                cerr << "Error: " << e.what() << endl;
            }
            
            auto end = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(end - start);
            long sleep_time = custom_spf - dur.count();
            if (sleep_time > 2) {
                this_thread::sleep_for(milliseconds(sleep_time));
            }
        }
        
        running = false;
        {
            lock_guard<mutex> lock(camera_mutex);
            frame_available.notify_all();
            queue_not_full.notify_all();
        }
        
        if (camera_thread.joinable()) {
            camera_thread.join();
        }
        
        destroyAllWindows();
        cout << "Multi-thread mode completed. Total frames: " << stats.total_frames << endl;
        
    } else {
        run_single_thread_mode(source);
    }
    
    return 0;
}