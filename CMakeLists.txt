cmake_minimum_required(VERSION 3.7)
project(gmu)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

set(SOURCE_FILES src/main.cpp src/HiddenMarkovModel.cpp src/HiddenMarkovModel.h
        src/Helpers.cpp src/Helpers.h src/Array.h)
add_executable(hmm ${SOURCE_FILES})
target_link_libraries(hmm ${OpenCL_LIBRARIES})