cmake_minimum_required(VERSION 3.7)
project(gmu)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCE_FILES src/HiddenMarkovModel.cpp src/HiddenMarkovModel.h
        src/Helpers.cpp src/Helpers.h src/Array.h src/HMMAlgorithm.h
        src/ForwardAlgorithm.cc src/ForwardAlgorithm.h src/ViterbiAlgorithm.cc
        src/ViterbiAlgorithm.h
        src/GPUImplementation.h src/GPUImplementation.cc
        src/ForwardAlgorithmGPU.cc src/ForwardAlgorithmGPU.h
        src/ViterbiAlgorithmGPU.cc src/ViterbiAlgorithmGPU.h
        src/oclHelpers.cc src/oclHelpers.h)
add_executable(hmm src/main.cc ${SOURCE_FILES})
target_link_libraries(hmm ${OpenCL_LIBRARIES})
