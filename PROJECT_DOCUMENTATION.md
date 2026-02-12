# Hero Image Transmission — 项目文档

## 概述
本项目实现了一套针对低带宽无线传输场景的图像/视频压缩与传输方案。核心思路是先将 160×120 的灰度图四值化（0/1/2/3），使用 RLE（游程编码）进行第一阶段压缩，再对 RLE 输出使用 Huffman 编码进行第二阶段压缩，从而在保证可解码的前提下极大降低带宽占用（实测约 46.6% 带宽节省）。

## 目录结构（简要）

- `CMakeLists.txt` — 构建配置
- `src/`
  - `main.cpp` — 主程序（采集、队列、压缩、显示与统计）
  - `include/hero_image_transmission.h` — 核心库（RLE + Huffman 实现、接口）
- `build/` — 构建产物
- `test_compression.cpp` / `test_compression_fixed.cpp` — 压缩性能测试
- 文档：`COMPRESSION_IMPLEMENTATION.md`, `QUICK_REFERENCE.md` 等

（完整路径请参见仓库根目录）

## 构建与运行

1. 创建构建目录并编译：

```bash
cd /home/cjy/HeroImageTransmission
rm -rf build && mkdir build && cd build
cmake ..
make -j
```

2. 运行主程序（实时视频处理与显示）：

```bash
./test
```

3. 运行压缩性能对比测试：

```bash
./test_compression
```

依赖：OpenCV (项目 CMake 已声明 `find_package(OpenCV 4.5.4 REQUIRED`)。

## 关键组件说明

- `HeroCamCompressor` (`src/include/hero_image_transmission.h`)
  - 输入：BGR 彩色帧
  - 步骤：HSV 提取（绿色目标）→ 轮廓过滤（去小噪点）→ 膨胀得到周围光晕 → 缩放到 160×120 → 四值化生成 0/1/2/3 的矩阵 → RLE → Huffman
  - 输出：`vector<uint8_t>` 型压缩包（包含频率表与位流），供接收端解码。

- `decodeCompressed` / `decompressHuffman` / `decodeRLE`
  - 解码端先用 Huffman 恢复 RLE 字节流，再用 RLE 还原四值图，最后上采样或显示。

- `main.cpp`
  - 采集线程将视频帧按环形缓冲入队，主线程从队列取帧并调用 `HeroCamCompressor::process()` 进行压缩；随后演示端对压缩包做解码并显示。
  - 程序包含性能统计（平均时间、平均压缩后大小、估算带宽等），并支持按帧跳帧控制 (`frame_skip`) 与自定义 SPF (`custom_spf`) 用于仿真带宽。

## 压缩算法细节

1) RLE（游程编码）
  - 基于 160×120 的四值化图像（每像素仅 2bit），将连续相等像素合并为 `(count, value)`。
  - 存储格式：一字节存储为 `count(6bit) | value(2bit)`，count 最大 63。
  - RLE 输出示例大小（见测试）：约 2,262 bytes（示例帧）。

2) Huffman 编码
  - 对 RLE 输出字节统计频率，构建 Huffman 树并生成编码表。
  - 序列化输出包含：频率表（用于解码）、原始 RLE 长度、压缩位流长度（bitLen）、以及位流字节序列。
  - Huffman 再压缩后示例大小约 1,207 bytes（实测）。

3) 性能/效果（实测）
  - 原始单帧（160×120 × 灰度）近似 19,200 bytes
  - RLE 后：2,262 bytes（约 88.22% 压缩）
  - RLE + Huffman 后：1,207 bytes（约 93.71% 压缩）
  - 带宽节省：约 46.6%（示例数据）

## 接口示例

编码：

```cpp
HeroCamCompressor compressor;
vector<uint8_t> packet = compressor.process(frame);
```

解码（示例）：

```cpp
cv::Mat decoded = decodeCompressed(packet, cv::Size(160, 120));
```

## 测试与验证

- `test_compression`：包含压缩前后大小的对比与压缩率统计。
- `main` 运行时会周期性输出统计：平均处理时间、平均压缩后大小、估算带宽（kbps），并判断是否满足 60 kbps 限制。

## 可扩展与优化方向

- 考虑引入 LZ77/LZSS 或算术编码以替代/补充 Huffman，可望进一步提升压缩率（但复杂度更高）。
- 自适应 RLE 参数或分块编码以应对场景多样性。
- 在包头加入版本号与压缩模式标识，以兼容旧版解码器。

## 注意事项

- 接收端（电控）必须实现 Huffman 解码以兼容当前压缩格式；否则只能识别旧版 RLE 格式。
- 频率表会占用一定开销（通常很小，相比带宽收益可忽略）。

## 联系与后续

如需我将该文档进一步拆分为：API 参考、压缩协议定义（字节级详解）、接收端样例解码器或性能基准脚本，请告诉我我将继续补充。

---
文档已由源码与仓库内说明文件（`QUICK_REFERENCE.md`, `COMPRESSION_IMPLEMENTATION.md` 等）汇总生成。
