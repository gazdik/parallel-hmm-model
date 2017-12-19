cmake_minimum_required(VERSION 3.7)
project(gmu)

set(CMAKE_CXX_STANDARD 14)

#find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

set(SOURCE_FILES src/main.cpp src/HiddenMarkovModel.cpp src/HiddenMarkovModel.h
        src/Helpers.cpp src/Helpers.h src/Array.h src/HMMAlgorithm.h
        src/ForwardAlgorithm.cc src/ForwardAlgorithm.h src/ViterbiAlgorithm.cc
        src/ViterbiAlgorithm.h src/BaumWelchAlgorithm.cc src/BaumWelchAlgorithm.h
        src/GPUImplementation.cc src/GPUImplementation.h)
add_executable(hmm ${SOURCE_FILES})
target_link_libraries(hmm ${OpenCL_LIBRARIES})
