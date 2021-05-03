cmake_minimum_required(VERSION 3.1)
project(PoseEstimation)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(pose_estimation pose_estimation.cpp)
target_link_libraries(pose_estimation ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SLAM} 
                      ${OpenCV_LIBS} Sophus::Sophus)