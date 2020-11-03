cmake_minimum_required(VERSION 3.1)

project(Feature_Extraction_Matching)

find_package(OpenCV 4 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(orb orb.cpp)
target_link_libraries(orb ${OpenCV_LIBS})