#ifndef THREAD_H
#define THREAD_H

#include "header.h"
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>

// ============ 环形缓冲区声明 ============
class RingBuffer {
public:
    explicit RingBuffer(int capacity);
    bool push(cv::Mat&& frame);
    bool pop(cv::Mat& frame);
    int size() const;
    int capacity() const;
    bool empty() const;
    bool full() const;

private:
    int capacity_;
    std::atomic<int> size_;
    int head_;
    int tail_;
    std::vector<cv::Mat> buffer_;
};

// ============ 全局变量声明（多线程相关） ============
extern RingBuffer frame_queue;
extern std::mutex camera_mutex;
extern std::condition_variable frame_available;
extern std::condition_variable queue_not_full;
extern std::atomic<bool> running;
extern int frame_skip;  // 跳帧参数：处理每第N帧（1表示不跳）

// ============ 性能统计结构声明 ============
struct PerfStats {
    std::vector<long> frame_times;
    std::vector<int> compressed_sizes;
    int total_frames = 0;
    int log_interval = 30;  // 每30帧输出一次统计
};

// ============ 线程函数声明 ============
void run_single_thread_mode(const std::string& source);
int camera_thread_func(const std::string& source);

#endif // THREAD_H