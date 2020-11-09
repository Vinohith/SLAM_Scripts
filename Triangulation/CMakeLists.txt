cmake_minimum_required(VERSION 3.1)
project(Triangulation)

find_package(OpenCV 4 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})