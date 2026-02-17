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
RingBuffer frame_queue(100);  // 增加缓冲区大小，避免帧丢失
std::mutex camera_mutex;
std::condition_variable frame_available;
std::condition_variable queue_not_full;
std::atomic<bool> running{true};
int frame_skip = 1;  // 跳帧参数：处理每第N帧（1表示不跳）

// 性能统计
struct PerfStats {
    std::vector<long> frame_times;
    std::vector<int> compressed_sizes;
    int total_frames = 0;
    int log_interval = 30;  // 每30帧输出一次统计
};

// 单线程模式：用于视频文件（简洁高效）
void run_single_thread_mode(const std::string& source) {
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << source << std::endl;
        return;
    }
    
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 30.0;
    std::cout << "Video FPS: " << fixed << setprecision(2) << video_fps << std::endl;
    
    long custom_spf = (long)(1000.0 / video_fps);
    
    HeroCamCompressor compressor;
    cv::Mat frame;
    ProcessResult result;
    
    const std::string window_name = "Operator View";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 480);
    
    // 预分配显示用的 Mat 对象（调整为 1280x480，用于并排显示）
    cv::Mat displayImg;
    displayImg.create(480, 1280, CV_8UC3);
    
    // 统计变量
    int total_frames = 0;
    std::vector<int> original_frame_sizes, compressed_frame_sizes;
    std::vector<long> frame_times;
    auto last_log_time = high_resolution_clock::now();
    
    while (cap.read(frame)) {
        if (frame.empty()) continue;
        
        auto start = high_resolution_clock::now();
        
        try {
            result = compressor.process(frame);
            
            if (!result.visualization.empty() && !result.compressData.empty()) {
                // 解压缩数据
                cv::Mat decompressed = decodeRLE(result.compressData, compressor.TARGET_SIZE);
                
                // 转换为 BGR 并上采样用于显示
                cv::Mat decompressed_bgr;
                cv::cvtColor(decompressed, decompressed_bgr, cv::COLOR_GRAY2BGR);
                cv::Mat decompressed_display;
                cv::resize(decompressed_bgr, decompressed_display, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
                
                // 原始轮廓图缩放到左侧
                cv::Mat original_display;
                cv::resize(result.visualization, original_display, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
                
                // 并排显示：左边原始轮廓，右边解压重建
                displayImg.create(480, 1280, CV_8UC3);
                original_display.copyTo(displayImg(cv::Rect(0, 0, 640, 480)));
                decompressed_display.copyTo(displayImg(cv::Rect(640, 0, 640, 480)));
                
                cv::imshow(window_name, displayImg);
            }
            
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
            
            total_frames++;
            original_frame_sizes.push_back(result.visualization.total() * result.visualization.elemSize());
            compressed_frame_sizes.push_back(result.compressData.size());
            frame_times.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - start).count());
            
            // 定期输出统计信息
            auto now = high_resolution_clock::now();
            if (duration_cast<seconds>(now - last_log_time).count() >= 5) {
                double fps = 1000.0 / custom_spf;
                int avg_orig_size = std::accumulate(original_frame_sizes.begin(), original_frame_sizes.end(), 0) / 
                                   (original_frame_sizes.empty() ? 1 : original_frame_sizes.size());
                int avg_comp_size = std::accumulate(compressed_frame_sizes.begin(), compressed_frame_sizes.end(), 0) / 
                                   (compressed_frame_sizes.empty() ? 1 : compressed_frame_sizes.size());
                long avg_time = std::accumulate(frame_times.begin(), frame_times.end(), 0L) / 
                               (frame_times.empty() ? 1 : frame_times.size());
                double bandwidth_original = (avg_orig_size * fps * 8.0) / 1000.0;
                double bandwidth_compressed = (avg_comp_size * fps * 8.0) / 1000.0;
                double compression_ratio = (1.0 - (double)avg_comp_size / avg_orig_size) * 100.0;
                
                cout << "\n[Frame " << total_frames << "] ===== STATISTICS =====" << endl;
                cout << "FPS: " << fixed << setprecision(1) << fps << " fps" << endl;
                cout << "Original Frame Size: " << avg_orig_size / 1024.0 << fixed << setprecision(2) << " KB" << endl;
                cout << "Compressed Frame Size: " << avg_comp_size / 1024.0 << fixed << setprecision(2) << " KB" << endl;
                cout << "Compression Ratio: " << fixed << setprecision(1) << compression_ratio << "%" << endl;
                cout << "Avg Process Time: " << avg_time << " ms" << endl;
                cout << "Bandwidth (Original): " << fixed << setprecision(2) << bandwidth_original << " kbps" << endl;
                cout << "Bandwidth (Compressed): " << fixed << setprecision(2) << bandwidth_compressed << " kbps" << endl;
                cout << "========================" << endl;
                
                last_log_time = now;
                frame_times.clear();
                original_frame_sizes.clear();
                compressed_frame_sizes.clear();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing frame: " << e.what() << std::endl;
        }
        
        // 帧率控制
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<milliseconds>(end - start);
        long sleep_time = custom_spf - dur.count();
        
        if (sleep_time > 2) {
            std::this_thread::sleep_for(milliseconds(sleep_time));
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    cout << "Single-thread mode completed. Total frames: " << total_frames << endl;
}


// 多线程模式：用于摄像头（防止丢帧）
int camera_thread_func(const std::string& source) {
    cv::VideoCapture cap;
    double source_fps = 30.0;  // 默认帧率
    
    // 判断是否从摄像头或者文件读取
    if (source == "0") {
        cap.open(0);  // 打开默认摄像头
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera" << std::endl;
            return -1;
        }
        // 设置摄像头缓冲大小为 1，减少延迟
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        source_fps = cap.get(cv::CAP_PROP_FPS);
    } else {
        cap.open(source);  // 打开视频文件
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << source << std::endl;
            return -1;
        }
        source_fps = cap.get(cv::CAP_PROP_FPS);
        std::cout << "Video FPS: " << source_fps << std::endl;
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
            
            // 等待队列不满（而不是丢弃帧）
            queue_not_full.wait(lock, []{return !running || !frame_queue.full();});
            
            if (!running) break;
            
            // 使用 move 语义避免拷贝
            if (!frame_queue.push(std::move(frame))) {
                std::cerr << "Failed to push frame to queue" << std::endl;
            }
            
            lock.unlock();
            frame_available.notify_one();
        }
        
        frame_count++;
        // 不睡眠，让摄像头线程尽快读取所有帧}
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
    std::cout << "1. Camera (Multi-thread mode) (press 1)" << std::endl;
    std::cout << "2. Video File (Single-thread mode) (press 2)" << std::endl;
    std::cout << "Please select (1 or 2): ";
    
    char choice;
    std::cin >> choice;
    
    std::string source;
    bool use_camera = false;
    
    if (choice == '1') {
        source = "0";
        use_camera = true;
        std::cout << "Using camera as source (Multi-thread mode)..." << std::endl;
    } else {
        source = "../vid/test_video1.mp4";
        std::cout << "Using video file (Single-thread mode): " << source << std::endl;
    }
    
    std::cout << std::endl;
    
    // ============ 根据数据源选择运行模式 ============
    if (use_camera) {
        // ============ 多线程模式（摄像头）============
        std::cout << "[Multi-thread mode] Starting..." << std::endl;
        running = true;  // 重置运行标志
        
        std::thread camera_thread(camera_thread_func, source);
        
        HeroCamCompressor compressor;
        ProcessResult result;
        long custom_spf = 33;
        
        const std::string window_name = "Operator View";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 1280, 480);
        
        cv::Mat displayImg;
        displayImg.create(480, 1280, CV_8UC3);
        
        PerfStats stats;
        auto last_log_time = high_resolution_clock::now();
        
        while(running) {
            auto start = high_resolution_clock::now();
            
            Mat frame;
            {
                std::unique_lock<std::mutex> lock(camera_mutex);
                
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
            
            try {
                result = compressor.process(frame);
                
                if (!result.visualization.empty() && !result.compressData.empty()) {
                    // 解压缩数据
                    cv::Mat decompressed = decodeRLE(result.compressData, compressor.TARGET_SIZE);
                    
                    // 转换为 BGR 并上采样用于显示
                    cv::Mat decompressed_bgr;
                    cv::cvtColor(decompressed, decompressed_bgr, cv::COLOR_GRAY2BGR);
                    cv::Mat decompressed_display;
                    cv::resize(decompressed_bgr, decompressed_display, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
                    
                    // 原始轮廓图缩放到左侧
                    cv::Mat original_display;
                    cv::resize(result.visualization, original_display, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
                    
                    // 并排显示：左边原始轮廓，右边解压重建
                    displayImg.create(480, 1280, CV_8UC3);
                    original_display.copyTo(displayImg(cv::Rect(0, 0, 640, 480)));
                    decompressed_display.copyTo(displayImg(cv::Rect(640, 0, 640, 480)));
                    
                    cv::imshow(window_name, displayImg);
                }
                
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    running = false;
                    break;
                }
                
                stats.frame_times.push_back(duration_cast<milliseconds>(high_resolution_clock::now() - start).count());
                stats.compressed_sizes.push_back(result.compressData.size());
                stats.total_frames++;
                
                auto now = high_resolution_clock::now();
                if (duration_cast<seconds>(now - last_log_time).count() >= 5) {
                    double fps = 1000.0 / 33.0;  // 摄像头模式下假设30FPS
                    int avg_orig_size = result.visualization.total() * result.visualization.elemSize();
                    int avg_comp_size = std::accumulate(stats.compressed_sizes.begin(), stats.compressed_sizes.end(), 0) / 
                                      (stats.compressed_sizes.empty() ? 1 : stats.compressed_sizes.size());
                    long avg_time = std::accumulate(stats.frame_times.begin(), stats.frame_times.end(), 0L) / 
                                   (stats.frame_times.empty() ? 1 : stats.frame_times.size());
                    double bandwidth_original = (avg_orig_size * fps * 8.0) / 1000.0;
                    double bandwidth_compressed = (avg_comp_size * fps * 8.0) / 1000.0;
                    double compression_ratio = (1.0 - (double)avg_comp_size / avg_orig_size) * 100.0;
                    
                    cout << "\n[Frame " << stats.total_frames << "] ===== STATISTICS =====" << endl;
                    cout << "FPS: " << fixed << setprecision(1) << fps << " fps" << endl;
                    cout << "Original Frame Size: " << avg_orig_size / 1024.0 << fixed << setprecision(2) << " KB" << endl;
                    cout << "Compressed Frame Size: " << avg_comp_size / 1024.0 << fixed << setprecision(2) << " KB" << endl;
                    cout << "Compression Ratio: " << fixed << setprecision(1) << compression_ratio << "%" << endl;
                    cout << "Avg Process Time: " << avg_time << " ms" << endl;
                    cout << "Bandwidth (Original): " << fixed << setprecision(2) << bandwidth_original << " kbps" << endl;
                    cout << "Bandwidth (Compressed): " << fixed << setprecision(2) << bandwidth_compressed << " kbps" << endl;
                    cout << "========================" << endl;
                    
                    last_log_time = now;
                    stats.frame_times.clear();
                    stats.compressed_sizes.clear();
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
            
            auto end = high_resolution_clock::now();
            auto dur = duration_cast<milliseconds>(end - start);
            long sleep_time = custom_spf - dur.count();
            
            if (sleep_time > 2) {
                std::this_thread::sleep_for(milliseconds(sleep_time));
            }
        }
        
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
        cout << "Multi-thread mode completed. Total frames: " << stats.total_frames << endl;
        
    } else {
        // ============ 单线程模式（视频文件）============
        std::cout << "[Single-thread mode] Starting..." << std::endl;
        run_single_thread_mode(source);
    }
    
    return 0;
}