cmake_minimum_required(VERSION 3.5)

project(apriltagWebcamTest)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(apriltagWebcamTest main.cpp)

target_link_libraries(apriltagWebcamTest ${OpenCV_LIBS} ${FREENECT_LIBRARIES} "/usr/local/lib/libapriltag.so" Eigen3::Eigen)
