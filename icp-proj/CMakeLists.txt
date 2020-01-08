cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pc_visualize)
find_package(PCL 1.9 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(icp icp.cpp)
target_link_libraries(icp ${PCL_LIBRARIES} ${OpenCV_LIBS})

