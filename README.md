# ModelCompression

[가짜연구소](https://pseudo-lab.com) 5기 실전경량화 스터디 repository 입니다. <br/>
딥러닝 모델 경량화 기술을 스터디하고, 다양한 platform 적용하여 이론과 실무 지식을 동시에 공부하는 것이 우리 스터디의 목적입니다. 

## 1. Install Dependency

### For x86/C++ Environments

**(1) TFLite**

- TBU

**(2) SNPE**

- TBU



### For aarch64 (e.g. Raspberry Pi) Environments

**(0) aarch64 gcc**

- TBU

**(1) TFLite**

- TBU

**(2) SNPE**

- TBU

**(3) OpenCV**

- TBU


## 2. Download models

- TBU

| Task           | Target | Quantization | Pruning | Files | Note |
|----------------|--------|:----:|:---:|:-----:|---|
| Classification | TFLite | -    | -   |       |   |
| Classification | TFLite | -    | 0.9 |       |   |
| Segmentation   | TFLite | -    | -   |       |   |
| Segmentation   | TFLite | int8 | -   |       |   |

## 3. Build and Run

### Build C++ interpreter
 
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

### Classification Example

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

## 4. Evaluation Result

### DeepLabv3


## Contributor

- 송태엽 _Taeyup Song_ | [Github](https://github.com/jerogar) |
- 선동진 _Dongjin Sun_ | [Github](-) |
- 고동근 _Donggeun Ko_ | [Github](-) |
- 이종호 | Github | 
- 오대환 _Daehwan Oh_ | [Github](-) | 
- 장진우 _Jinwoo Jang_ | [Github](https://github.com/Jinwoo1126) | 
- 손인석 _Inseok Son_ | [Github](https://github.com/inseokson) | 
- 박수범 _Subeom Park_ | [Github](-) | 
- 김용환 _Yonghwan Kim_ | [Github](https://github.com/yonghwan1994) | 
- 김서영 | Github | 
- 신세별 | Github | 
- 박명규 _Myung Gyu Park_ | [Github](https://github.com/audrb1999) | 
- 이은식 _Eunsik Lee_ | [Github](https://github.com/emphasis10) | 
- 김정현 _Jeonghyon Kim_ | [Github](https://github.com/kimjeonghyon) | 
