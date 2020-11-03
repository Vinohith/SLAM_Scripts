cmake_minimum_required(VERSION 3.1)
project(Pose_Estimation)

find_package(OpenCV 4 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(pose_2d2d pose_2d2d.cpp)
target_link_libraries(pose_2d2d ${OpenCV_LIBS})