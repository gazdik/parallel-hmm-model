cmake_minimum_required(VERSION 3.7)
project(gmu)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

set(SOURCE_FILES src/HiddenMarkovModel.cpp src/HiddenMarkovModel.h
        src/Helpers.cpp src/Helpers.h src/Array.h src/HMMAlgorithm.h
        src/ForwardAlgorithm.cc src/ForwardAlgorithm.h src/ViterbiAlgorithm.cc
        src/ViterbiAlgorithm.h src/BaumWelchAlgorithm.cc src/BaumWelchAlgorithm.h)
add_executable(hmm src/main.cpp ${SOURCE_FILES})

set(SOURCE_FILES_GPU src/GPUImplementation.h src/GPUImplementation.cc
        src/ForwardAlgorithmGPU.cc src/ForwardAlgorithmGPU.h
        src/BaumWelchAlgorithmGPU.cc src/BaumWelchAlgorithmGPU.h
        src/ViterbiAlgorithmGPU.cc src/ViterbiAlgorithmGPU.h)
add_executable(hmmGPU src/mainGPU.cc ${SOURCE_FILES} ${SOURCE_FILES_GPU})
target_link_libraries(hmmGPU ${OpenCL_LIBRARIES})