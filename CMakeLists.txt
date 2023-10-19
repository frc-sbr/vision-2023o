cmake_minimum_required(VERSION 3.5)

project(apriltagtest)

find_package(OpenCV REQUIRED)
find_package(libfreenect REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories("/usr/include/libusb-1.0/" ${EIGEN3_INCLUDE_DIR})

add_executable(apriltagtest main.cpp)

target_link_libraries(apriltagtest ${OpenCV_LIBS} ${FREENECT_LIBRARIES} "/usr/local/lib/libapriltag.so" Eigen3::Eigen)
