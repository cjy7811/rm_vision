cmake_minimum_required(VERSION 3.10)
project(hero_image_transmission)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV 4.5.4 REQUIRED)

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src/include)

# 可执行文件：主程序
add_executable(test
    src/main.cpp
)

# 链接OpenCV库
target_link_libraries(test
    ${OpenCV_LIBS}
    pthread
)

# 设置编译器标志
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(test PRIVATE -Wall -O2 -march=native)
endif()