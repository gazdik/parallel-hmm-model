cmake_minimum_required(VERSION 3.7)
project(gmu)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/main.cpp src/HiddenMarkovModel.cpp src/HiddenMarkovModel.h
        src/Helpers.cpp src/Helpers.h)
add_executable(hmm ${SOURCE_FILES})