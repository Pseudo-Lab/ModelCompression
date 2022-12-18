cmake_minimum_required(VERSION 3.16)
project(Evaluation)

unset(CMAKE_BUILD_TYPE CACHE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_AARCH64 "build for aarch64" OFF) # BUILD_BOARD_MODE
option(USE_TFLITE "Build TFLite" ON)
option(USE_SNPE "Build SNPE" OFF)

message(STATUS "====================================================================")
message(STATUS "AARCH64 build: " ${BUILD_AARCH64})
message(STATUS "TFLite: " ${USE_TFLITE})
message(STATUS "SNPE: " ${USE_SNPE})
message(STATUS "====================================================================")

# Set OpenCV install Path
if( "${OPENCV_PATH}" STREQUAL "" )
    if (BUILD_AARCH64)
        set(OPENCV_PATH /opt/opencv412_aarch64/ CACHE PATH "Set OpenCV Install Path")
    else()
        set(OPENCV_PATH /usr/local CACHE PATH "Set OpenCV Install Path")
    endif()
endif()
set(OPENCV_INSTALL_PATH ${OPENCV_PATH})
message(STATUS "OpenCV install Path: " ${OPENCV_INSTALL_PATH})
unset(OPENCV_PATH CACHE)
message(STATUS "====================================================================")

# Set DNN Lib. Path
if(USE_SNPE)
    if( "${SNPE_PATH}" STREQUAL "" )
        set(SNPE_PATH "/opt/snpe-1.66.0.3729/" CACHE STRING "SNPE install path")
    endif()
    set(SNPE_ROOT ${SNPE_PATH})
    if (BUILD_AARCH64)
        set(SNPE_LIB_ROOT ${SNPE_ROOT}/lib/aarch64-ubuntu-gcc7.5)
    else()
        set(SNPE_LIB_ROOT ${SNPE_ROOT}/lib/x86_64-linux-clang)
    endif()
    unset(SNPE_PATH CACHE)


    message(STATUS "SNPE install Path: " ${SNPE_ROOT})
    message(STATUS "SNPE Lib. Path: " ${SNPE_LIB_ROOT})
    message(STATUS "====================================================================")
endif()

if(USE_TFLITE)
    if( "${TF_PATH}" STREQUAL "" )
        set(TF_PATH "/opt/tensorflow_src" CACHE STRING "TFLite install path")
    endif()
    set(TFLITE_ROOT ${TF_PATH})
    unset(TF_PATH CACHE)
    message(STATUS "TensorFlow Src. Path: " ${TFLITE_ROOT})


    if( "${TFLITE_LIB_PATH}" STREQUAL "" )
        if (BUILD_AARCH64)
            set(TFLITE_LIB_PATH "/opt/tflite_r2.10_cpp_lib_aarch64" CACHE STRING "TFLite lib. path")
        else()
            set(TFLITE_LIB_PATH "/opt/tflite_r2.10_cpp_lib_x86" CACHE STRING "TFLite lib. path")
        endif()
    endif()
    set(TFLITE_LIB_ROOT ${TFLITE_LIB_PATH})
    unset(TFLITE_LIB_PATH CACHE)
    message(STATUS "TFLite Lib. Path: " ${TFLITE_LIB_ROOT})

    if( "${FLATBUFFER_PATH}" STREQUAL "" )
        set(FLATBUFFER_PATH "/opt/flatbuffers" CACHE STRING "flatbuffer lib. path")
    endif()
    set(FLATBUFFER_ROOT ${FLATBUFFER_PATH})
    unset(FLATBUFFER_PATH CACHE)
    message(STATUS "FlatBuffer install Path: " ${FLATBUFFER_ROOT})
endif()
message(STATUS "====================================================================")

# compiler option
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s")

if (BUILD_AARCH64)

    message(STATUS "Compiler: aarch64-linux-gnu-gcc, aarch64-linux-gnu-g++")
    add_compile_definitions(BOARD)

    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lm -lrt -fPIC")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lm -lrt -fPIC -march=armv8-a")
endif()


set(RUNTIME_SRC_PATH Interpreter/IInterpreter.cpp Interpreter/DnnInference.cpp)
if (USE_TFLITE)
    add_compile_definitions(TFLite_RUNTIME)

    include_directories(${TFLITE_ROOT}/)
    include_directories(${FLATBUFFER_ROOT}/include)
    LINK_DIRECTORIES(${TFLITE_LIB_ROOT})

    set(RUNTIME_LIB_PACKAGE ${RUNTIME_LIB_PACKAGE} tensorflowlite)
    set(RUNTIME_SRC_PATH ${RUNTIME_SRC_PATH} Interpreter/TfLiteInference.cpp)
endif()

if(USE_SNPE)
    add_compile_definitions(SNPE_RUNTIME)

    include_directories(${SNPE_ROOT}/include/zdl)
    link_directories(${SNPE_LIB_ROOT})

    set(RUNTIME_LIB_PACKAGE ${RUNTIME_LIB_PACKAGE} SNPE)
    set(RUNTIME_SRC_PATH ${RUNTIME_SRC_PATH} Interpreter/SnpeInference.cpp)
endif()


# ADD source code
set(TASK_SRC_PATH DeepLabv3/DeepLabv3.cpp)
set(TASK_SRC_PATH ${TASK_SRC_PATH} Classification/Classification.cpp Classification/MnistEval.cpp Classification/CifarEval.cpp)

include_directories(${OPENCV_INSTALL_PATH}/include/opencv4)
include_directories(${OPENCV_INSTALL_PATH}/include/opencv4/opencv2)
link_directories(${OPENCV_INSTALL_PATH}/lib)
set(OPENCV_LIB_PACKAGE opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_dnn opencv_videoio)

add_executable(Evaluation main.cpp ${TASK_SRC_PATH} ${RUNTIME_SRC_PATH})
target_link_libraries(Evaluation rt ${OPENCV_LIB_PACKAGE} ${RUNTIME_LIB_PACKAGE})

unset(USE_TFLITE CACHE)
unset(USE_SNPE CACHE)
