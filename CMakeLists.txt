cmake_minimum_required(VERSION 3.1.5)

project(resize)

set(CMAKE_BUILD_TYPE"Release")

set(CMAKE_CXX_STANDARD 11)
add_compile_options(-msse4.1) 

# set(OpenCV_DIR /home/zd/3rdparty/opencv-4.6.0/install/lib/cmake/opencv4)
find_package(OpenCV 4.6.0 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(run main.cpp)

target_link_libraries(run 
    ${OpenCV_LIBS}
)
