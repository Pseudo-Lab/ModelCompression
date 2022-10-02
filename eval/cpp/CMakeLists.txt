cmake_minimum_required(VERSION 3.16)
project(DeepLabv3)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(BUILD_BOARD_MODE 0)
set(BUILD_TFLITE_MODE 0)
set(BUILD_SNPE_MODE 1)

# Set OpenCV Path
if (${BUILD_BOARD_MODE})
    set(OPENCV_INSTALL_PATH /media/Data/lib/opencv412/platforms/linux/build/)
else()
    set(OPENCV_INSTALL_PATH /usr/local/)

endif()

# Set DNN Lib. Path
set(SNPE_ROOT /media/Data/lib/snpe-1.66.0.3729/)
set(TFLITE_ROOT /media/Data/lib/tensorflow_src_rb5_r2.5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s")


if (${BUILD_BOARD_MODE})

    message(STATUS "Build Mode: BOARD")
    message(STATUS "Compiler: aarch64-linux-gnu-gcc, aarch64-linux-gnu-g++")

    add_compile_definitions(BOARD)

    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lm -lrt -fPIC -march=armv8-a")

    if (${BUILD_TFLITE_MODE})

        message(STATUS "Runtime: TFLite")
        add_compile_definitions(TFLite_RUNTIME)

        ## r2.5 version
        include_directories(${TFLITE_ROOT})
        include_directories(${TFLITE_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include/)

        set(TFLITE_BUILD_ROOT ${TFLITE_ROOT}/tensorflow/lite/tools/make/)

        ADD_LIBRARY(TFlite_LIB STATIC IMPORTED)
        SET_TARGET_PROPERTIES(TFlite_LIB PROPERTIES IMPORTED_LOCATION ${TFLITE_BUILD_ROOT}/gen/linux_aarch64/lib/libtensorflow-lite.a )

    elseif(${BUILD_SNPE_MODE})
        message(STATUS "Runtime: SNPE")
        add_compile_definitions(SNPE_RUNTIME)

        include_directories(${SNPE_ROOT}/include/zdl)
        link_directories(${SNPE_ROOT}/lib/aarch64-ubuntu-gcc7.5)

    else()
        message(STATUS "Runtime: OpenCV")

    endif()

else()
    message(STATUS "Build Mode: PC")

    if (${BUILD_TFLITE_MODE})

        message(STATUS "Runtime: TFLite")
        add_compile_definitions(TFLite_RUNTIME)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lrt")
        set(TFLITE_BUILD_PATH "/media/Data/lib/tensorflow_src")

        set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
                "Directory that contains the TensorFlow project" )
        if(NOT TENSORFLOW_SOURCE_DIR)
            get_filename_component(TENSORFLOW_SOURCE_DIR
                    "${TFLITE_BUILD_PATH}" ABSOLUTE)
        endif()

        add_subdirectory(
                "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
                "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)


    elseif(${BUILD_SNPE_MODE})

        message(STATUS "Runtime: SNPE")
        add_compile_definitions(SNPE_RUNTIME)

        include_directories(${SNPE_ROOT}/include/zdl)
        link_directories(${SNPE_ROOT}/lib/x86_64-linux-clang)

    else()
        message(STATUS "Runtime: OpenCV")
    endif()

endif()

if (${BUILD_TFLITE_MODE})
	set(RUNTIME_LIB_PACKAGE tensorflow-lite)
	set(RUNTIME_SRC_PATH TFLite)
elseif(${BUILD_SNPE_MODE})
    set(RUNTIME_LIB_PACKAGE SNPE)
    set(RUNTIME_SRC_PATH SNPE)
else()
    set(RUNTIME_SRC_PATH OpenCV)
endif()

file(GLOB_RECURSE RUNTIME_DIR CONFIGURE_DEPENDS ${RUNTIME_SRC_PATH}/*.cpp ${RUNTIME_SRC_PATH}/*.h)
file(GLOB_RECURSE DNNCOMMON_DIR CONFIGURE_DEPENDS src/dnn/common/*.cpp src/dnn/common/*.h)

include_directories(${OPENCV_INSTALL_PATH}/include/opencv4)
include_directories(${OPENCV_INSTALL_PATH}/include/opencv4/opencv2)
link_directories(${OPENCV_INSTALL_PATH}/lib)
set(OPENCV_LIB_PACKAGE opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_dnn opencv_videoio)

#add_executable(DeepLabv3 ${FACEDET_DIR} ${RUNTIME_DIR} ${DNNCOMMON_DIR} src/main.cpp)
add_executable(DeepLabv3 main.cpp ${RUNTIME_DIR} DeepLabv3.cpp)
target_link_libraries(DeepLabv3 rt ${OPENCV_LIB_PACKAGE} ${RUNTIME_LIB_PACKAGE})

