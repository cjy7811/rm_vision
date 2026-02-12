#include "hero_image_transmission.h"
#include <chrono>
#include <thread>
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <iomanip>
#include <numeric>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ============ 环形缓冲区 ============
class RingBuffer {
public:
    explicit RingBuffer(int capacity) : capacity_(capacity), size_(0), head_(0), tail_(0),
                                       buffer_(capacity) {}
    
    bool push(Mat&& frame) {
        if (size_ >= capacity_) {
            return false;  // 缓冲区满
        }
        buffer_[tail_] = std::move(frame);
        tail_ = (tail_ + 1) % capacity_;
        size_++;
        return true;
    }
    
    bool pop(Mat& frame) {
        if (size_ == 0) {
            return false;  // 缓冲区空
        }
        frame = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        size_--;
        return true;
    }
    
    int size() const { return size_; }
    int capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ >= capacity_; }
    
private:
    int capacity_;
    std::atomic<int> size_;
    int head_;
    int tail_;
    std::vector<Mat> buffer_;
};

// 全局变量
RingBuffer frame_queue(20);
std::mutex camera_mutex;
std::condition_variable frame_available;
std::condition_variable queue_not_full;
std::atomic<bool> running{true};
int frame_skip = 5;  // 跳帧参数：处理每第N帧（1表示不跳）

// 性能统计
struct PerfStats {
    std::vector<long> frame_times;
    std::vector<int> compressed_sizes;
    int total_frames = 0;
    int log_interval = 30;  // 每30帧输出一次统计
};

int camera_thread_func(const std::string& source) {
    cv::VideoCapture cap;
    
    // 判断是否从摄像头或者文件读取
    if (source == "0") {
        cap.open(0);  // 打开默认摄像头
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera" << std::endl;
            return -1;
        }
    } else {
        cap.open(source);  // 打开视频文件
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << source << std::endl;
            return -1;
        }
    }

    Mat frame;
    int frame_count = 0;
    
    while (running) {
        if (!cap.read(frame)) {
            break;  // 视频结束
        }
        
        if (frame.empty()) continue;
        
        // 跳帧逻辑
        if (frame_count % frame_skip != 0) {
            frame_count++;
            continue;
        }
        
        {
            std::unique_lock<std::mutex> lock(camera_mutex);
            
            // 等待队列不满
            queue_not_full.wait(lock, []{
                return !running || !frame_queue.full();
            });
            
            if (!running) break;
            
            // 使用 move 语义避免拷贝
            if (!frame_queue.push(std::move(frame))) {
                std::cerr << "Failed to push frame to queue" << std::endl;
            }
            
            lock.unlock();
            frame_available.notify_one();
        }
        
        frame_count++;
        std::this_thread::sleep_for(milliseconds(5));  // 减少睡眠时间
    }
    
    // 通知主线程摄像头线程已结束
    {
        std::lock_guard<std::mutex> lock(camera_mutex);
        running = false;
        frame_available.notify_all();
        queue_not_full.notify_all();
    }
    
    cap.release();
    return 0;
}

int main() {
    // 提示用户选择数据源
    std::cout << "=== Image Source Selection ===" << std::endl;
    std::cout << "1. Camera (press 1)" << std::endl;
    std::cout << "2. Video File (press 2)" << std::endl;
    std::cout << "Please select (1 or 2): ";
    
    char choice;
    std::cin >> choice;
    
    std::string source;
    if (choice == '1') {
        source = "0";  // 使用摄像头
        std::cout << "Using camera as source..." << std::endl;
    } else {
        source = "../vid/test_video1.mp4";  // 使用视频文件（相对路径）
        std::cout << "Using video file: " << source << std::endl;
    }
    
    std::cout << std::endl;
    
    // 创建摄像头线程
    std::thread camera_thread(camera_thread_func, source);
    
    // 初始化压缩器
    HeroCamCompressor compressor;
    vector<uint8_t> packet;
    
    // 帧率控制参数
    frame_skip = 6; // 调整为5，确保带宽不超过60kbps
    long custom_spf = 200;
    
    // 窗口名称和预分配显示缓冲区
    const std::string window_name = "Operator View";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 640, 480);
    
    // 预分配显示用的 Mat 对象，避免每帧都创建
    cv::Mat decodedImg, showImg;
    showImg.create(480, 640, CV_8UC1);
    
    // 性能统计
    PerfStats stats;
    
    while(running) {
        auto start = high_resolution_clock::now();
        
        // 从队列获取帧（使用 move 语义）
        Mat frame;
        {
            std::unique_lock<std::mutex> lock(camera_mutex);
            
            frame_available.wait_for(lock, milliseconds(100), []{
                return !running || !frame_queue.empty();
            });
            
            if (!running && frame_queue.empty()) {
                break;
            }
            
            if (frame_queue.empty()) {
                continue;
            }
            
            if (!frame_queue.pop(frame)) {
                continue;
            }
            
            lock.unlock();
            queue_not_full.notify_one();
        }
        
        try {
            if (frame.empty()) continue;
            
            // 压缩帧
            packet = compressor.process(frame);
            
            // 解码并显示（重用预分配的 Mat）
            decodedImg = decodeCompressed(packet, compressor.TARGET_SIZE);
            if (!decodedImg.empty()) {
                cv::resize(decodedImg, showImg, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
                showImg.convertTo(showImg, -1, 1.2, 10);
                cv::imshow(window_name, showImg);
            }
            
            // 检查用户输入
            char key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                running = false;
                break;
            }
            
            // 收集性能数据
            auto end = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(end - start);
            stats.frame_times.push_back(dur.count());
            stats.compressed_sizes.push_back(packet.size());
            stats.total_frames++;
            
            // 定期输出统计信息（每 N 帧）
            if (stats.total_frames % stats.log_interval == 0) {
                long avg_time = std::accumulate(stats.frame_times.begin(), stats.frame_times.end(), 0L) / stats.frame_times.size();
                int avg_size = std::accumulate(stats.compressed_sizes.begin(), stats.compressed_sizes.end(), 0) / stats.compressed_sizes.size();
                long max_time = *std::max_element(stats.frame_times.begin(), stats.frame_times.end());
                
                // 计算带宽：帧大小(byte) * 帧率(fps) * 8(bit/byte) / 1000(kbps转换)
                double fps = 1000.0 / custom_spf;  // 帧率 = 1000ms / 每帧毫秒
                double bandwidth_kbps = (avg_size * fps * 8) / 1000.0;
                
                cout << "[Frame " << stats.total_frames << "] "
                     << "Avg time: " << avg_time << "ms, "
                     << "Max time: " << max_time << "ms, "
                     << "Avg size: " << avg_size << " bytes, "
                     << "FPS: " << fixed << setprecision(1) << fps << ", "
                     << "Bandwidth: " << fixed << setprecision(2) << bandwidth_kbps << " kbps";
                
                // 判断是否满足60kbps限制
                if (bandwidth_kbps <= 60.0) {
                    cout << " ✓ (满足60kbps要求)";
                } else {
                    cout << " ✗ (超出60kbps要求 " << fixed << setprecision(2) << (bandwidth_kbps - 60.0) << "kbps)";
                }
                cout << endl;
                
                stats.frame_times.clear();
                stats.compressed_sizes.clear();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing frame: " << e.what() << std::endl;
        }
        
        // 帧率控制
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<milliseconds>(end - start);
        long sleep_time = custom_spf - dur.count();
        
        if (sleep_time > 0) {
            std::this_thread::sleep_for(milliseconds(sleep_time));
        }
    }
    
    // 清理资源
    running = false;
    
    {
        std::lock_guard<std::mutex> lock(camera_mutex);
        frame_available.notify_all();
        queue_not_full.notify_all();
    }
    
    if (camera_thread.joinable()) {
        camera_thread.join();
    }
    
    cv::destroyAllWindows();
    
    std::cout << "Program terminated successfully. Total frames processed: " << stats.total_frames << std::endl;
    return 0;
}