cmake_minimum_required(VERSION 3.0)
project(loop_closure)

# opencv 
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
 
set(DBoW3_INCLUDE_DIRS "/usr/local/include")
set(DBoW3_LIBS "/usr/local/lib/libDBoW3.so")

add_executable(bow_features bow_features.cpp)
target_link_libraries(bow_features ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(loop_closure_kitti loop_closure_kitti.cpp)
target_link_libraries(loop_closure_kitti ${OpenCV_LIBS} ${DBoW3_LIBS})