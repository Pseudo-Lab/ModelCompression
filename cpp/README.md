# TFLite and SNPE C++ Interpreter

## 1. Install Dependency

## (1) For x86/C++ Environments

### **TensorFlow Lite**

- Clone tensorflow repository
    ```bash
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    ```

- Add options `tflite_with_xnnpack`, `xnn_enable_qs8` for optimizing quantized/pruned model on CPU environments.

    ```bash
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow

    bazel build --define tflite_with_xnnpack=true --define xnn_enable_qs8=true -c opt //tensorflow/lite:libtensorflowlite.so
    ```

- Build flatbuffer lib. 
    ```bash
    cd {your lib. path}

    git clone https://github.com/google/flatbuffers.git
    cd flatbuffers
    
    cmake -G "Unix Makefiles"
    make
    make install
    ```


### **SNPE for Qualcomm AP.**

- TBU



### (2) For aarch64 (e.g. Raspberry Pi) Environments

### **aarch64 gcc**

```bash
sudo apt-get install gcc-aarch64-linux-gnu
sudo apt-get install g++-aarch64-linux-gnu
```

### **TensorFlow Lite**

- Build TFLite aarch64 shared library. (Teference: [Build TensorFlow Lite for ARM boards](https://www.tensorflow.org/lite/guide/build_arm#step_3_build_arm_binary))

  ```bash
  bazel build --config=elinux_aarch64 --define tflite_with_xnnpack=true --define xnn_enable_qs8=true -c opt //tensorflow/lite:libtensorflowlite.so
  ```

- copy `bazel-bin/tensorflow/lite/libtensorflowlite.so` to directory your own.

  ```bash
  # e.g. copy to "/opt/tflite_aarch64_lib"
  cp bazel-bin/tensorflow/lite/libtensorflowlite.so /opt/tflite_aarch64_lib/
  ```

### **SNPE for Qualcomm AP.**

- TBU

### **OpenCV.**

- Build OpenCV 4.1.2 for aarch64
    
    ```bash
    git clone https://github.com/opencv/opencv.git
    cd opencv
    git checkout 4.1.2
    cd ..
    
    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv_contrib
    git checkout 4.1.2
    cd ..
    ```

    ```bash
    cd opencv/platforms/linux
    mkdir -p build
    cd build
    ```
    ```bash
    cmake -DCMAKE_TOOLCHAIN_FILE=../aarch64-gnu.toolchain.cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_NEON=ON \
    ../../..
    ```
    ```bash
    make
    make install
    ```

- Find shared lib. in `opencv/platform/linux/build/install/lib`. 

## 2. Build
 
- clone this repository
    ```bash
    git clone https://github.com/Pseudo-Lab/ModelCompression.git
    cd ModelCompression/cpp
    ```

- build cmake project
    ```bash
    mkdir build
    cd build
    # cmake with default options
    cmake .. \
          -D USE_TFLITE=ON \
          -D OPENCV_PATH=/usr/local \
          -D TF_PATH=/opt/tensorflow_src \
          -D TFLITE_LIB_PATH=/opt/tflite_x86 \
          -D FLATBUFFER_PATH=/opt/flatbuffer \
          -D USE_SNPE=OFF \
          -D SNPE_PATH=/opt/snpe-1.66.0.3729
    ```
- If you need cross-compile for aarch64 AP
  ```bash
  mkdir build
  cd build
  # cmake with default options
  cmake .. \
        -D BUILD_AARCH64=ON \
        -D USE_TFLITE=ON \
        -D OPENCV_PATH=/usr/local \
        -D TF_PATH=/opt/tensorflow_src \
        -D TFLITE_LIB_PATH=/opt/tflite_x86 \
        -D FLATBUFFER_PATH=/opt/flatbuffer \
        -D USE_SNPE=OFF \
        -D SNPE_PATH=/opt/snpe-1.66.0.3729
  ```

  ```bash
  make
  ```

## 3. Run

- Example of MobileNet v1 

  ```bash
  ./Evaluation  -mode cls \
                -modelPath ../../models/MobileNet/mobilenet_v1_1.0_224.tflite \
                -inputScale 0.0039 \
                -inputMean 0 0 0 \
                -inputPath ../../datasets/Ryanair.jpg \
                -labelPath ../../datasets/imageNet_labels_1001.txt
  ```
  
  - Expected output
  ```
  Model Path: ../../models/mobilenet_v1_1.0_224.tflite
  image(DB) Path: ../../datasets/Ryanair.jpg
  label Path: ../../datasets/imageNet_labels.txt
  INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  Load complete: ../../models/mobilenet_v1_1.0_224.tflite
  -> tensors size: 103
  -> nodes size: 33
  -> inputs: 1
  --> input(0) name: input
  --> input(0) shape = [1, 224, 224, 3]
  -> Outputs: 1
  --> output(0) name: MobilenetV1/Predictions/Reshape_1
  --> output(0) Shape = [1, 1001]
  Top-0: (idx= 0, label= 405:airliner), prob.: 0.885527
  Top-1: (idx= 1, label= 909:wing), prob.: 0.084947
  Top-2: (idx= 2, label= 896:warplane, military plane), prob.: 0.0270763
  Top-3: (idx= 3, label= 813:space shuttle), prob.: 0.00147471
  Top-4: (idx= 4, label= 406:airship, dirigible), prob.: 0.000880025
  Eval. time (only inference): 28[msec]
  ```

- Example of pruned MobileNet model

  ```bash
  ./Evaluation  -mode cls \
                -modelPath ../../models/MobileNet/mbv1_100_90_12b4_684.tflite \
                -inputScale 1 \
                -inputMean 0 0 0 \
                -inputPath ../../datasets/Ryanair.jpg \
                -labelPath ../../datasets/imageNet_labels_1000.txt
  ```
  
  - Expected output
    ```
    Model Path: ../../models/MobileNet/mbv1_100_90_12b4_684.tflite
    image(DB) Path: ../../datasets/Ryanair.jpg
    label Path: ../../datasets/imageNet_labels_1000.txt
    INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
    Load complete: ../../models/MobileNet/mbv1_100_90_12b4_684.tflite
    -> tensors size: 136
    -> nodes size: 51
    -> inputs: 1
    --> input(0) name: float_image_input
    --> input(0) shape = [1, 224, 224, 3]
    -> Outputs: 1
    --> output(0) name: resnet_model/final_dense_1
    --> output(0) Shape = [1, 1000]
    Top-0: (idx= 0, label= 405:airliner), prob.: 9.05931
    Top-1: (idx= 1, label= 896:warplane, military plane), prob.: 7.84844
    Top-2: (idx= 2, label= 909:wing), prob.: 6.87741
    Top-3: (idx= 3, label= 406:airship, dirigible), prob.: 5.97151
    Top-4: (idx= 4, label= 813:space shuttle), prob.: 4.71733
    Eval. time (only inference): 24[msec]
    ```

- Example of DeepLabv3

  ```bash
  ./Evaluation  -mode seg \
                -modelPath ../../models/deeplabv3/tflite/deeplabv3_mnv2_dm05_pascal_trainval_fp32.tflite \
                -inputScale 0.007843 \
                -inputMean 127.5 127.5 127.5 \
                -inputPath ../../datasets/deeplab1.png
  ```

- Evaluation of PASCAL VOC Test dev.

  ```bash
  ./Evaluation  -mode pascal \
                -modelPath ../../models/deeplabv3/tflite/deeplabv3_mnv2_dm05_pascal_trainval_fp32.tflite \
                -inputScale 0.007843 \
                -inputMean 127.5 127.5 127.5 \
                -inputPath ../../datasets/VOCdevkit/VOC2012
  ```
