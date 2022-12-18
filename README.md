# ModelCompression

[가짜연구소](https://pseudo-lab.com) 5기 실전경량화 스터디 repository 입니다. <br/>
딥러닝 모델 경량화 기술을 스터디하고, 다양한 platform 적용하여 이론과 실무 지식을 동시에 공부하는 것이 우리 스터디의 목적입니다. 

## Blog

- [가짜연구소 5기 - 실전 경량화 스터디](https://chanrankim.notion.site/Model-Blog-6c2a7fc320104f79b383071a7ce8b4fc)

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

---

## 1. TFLite/SNPE/OpenCV C++ interpreter

- [https://github.com/Pseudo-Lab/ModelCompression/tree/main/cpp](https://github.com/Pseudo-Lab/ModelCompression/tree/main/cpp)

## 2. TFLite/SNPE/OpenCV Python interpreter

- [https://github.com/Pseudo-Lab/ModelCompression/tree/main/python](https://github.com/Pseudo-Lab/ModelCompression/tree/main/python)

## 3. Tools (convert, pruning, quantization)

- TBU

## 2. Download models


| Task           | Target | Quantization | Pruning | Files | Note |
|----------------|--------|:----:|:---:|:-----:|---|
| Classification | TFLite | -    | -   |  [original link (tf model zoo)](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)     |   |
| Classification | TFLite | -    | 0.9 |  [original link (google research)](https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_100_90_12b4_684.tflite)     | [repo](https://github.com/google-research/google-research/tree/master/fastconvnets)  |
| Segmentation   | TFLite | -    | -   |  [original link (tf model zoo)](http://download.tensorflow.org/models/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz)     |   |
| Segmentation   | TFLite | fp16    | -   | [google drive](https://drive.google.com/file/d/15cra3-phPmVxTrr2BsPWA4A-mQEzbBpv/view?usp=share_link)      |   |
| Segmentation   | TFLite | dynamic range  | -   | [google drive](https://drive.google.com/file/d/1MkHYyDcX_AVUX1CrrpCOga5xICNDvAaA/view?usp=share_link)     |   |
| Segmentation   | TFLite | int8 | -   | [google drive ](https://drive.google.com/file/d/1LZp6or_o3DpbtR-4eYp8LDdPQuxDHjxt/view?usp=share_link)  |   |
| Segmentation   | SNPE | - | -   | [google drive ](https://drive.google.com/file/d/1fYG_DQ8sIChEb_BTagaLEWDyDbzSc5i6/view?usp=share_link)  |   |

## 4. Evaluation Result

### (1) Classification

| No. | Platform | Lib. (Runtime) | Model | Pruning Method | Accuracy | Average Processing Time | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PC | TFLite (XNNPACK) | [mobilenet_v1_1.0_224.tflite](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | X | 0.9759 | 28 msec | 16.9 Mb |
| 2 | PC | TFLite (XNNPACK) | [mbv1_100_90_12b4_684.tflite](https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_100_90_12b4_684.tflite) | pruned (0.9) | 0.9684 | 24 msec | 6.2 Mb |
| 3 | RPI3 | TFLite (XNNPACK) | [mobilenet_v1_1.0_224.tflite](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | X | 0.9687 | 565 msec | 16.9 Mb |
| 4 | RPI3 | TFLite (XNNPACK) | [mbv1_100_90_12b4_684.tflite](https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_100_90_12b4_684.tflite) | strip_pruning (0.9) | 0.9692 | 309 msec | 6.2 Mb |



### (2) Semantic Segmentation: DeepLabv3

| No. | Platform | Lib. (Runtime) | Model | Quantization Method | Pruning Method | mIoU | Average Processing Time | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PC | TFLite(Python. XNNPACK) | [deeplabv3_mnv2_dm05_pascal_trainval_dynamic_fp32.tflite](https://drive.google.com/file/d/1uy17pDW-JLcMXLDpGACBKmKbrstK0oYx/view?usp=sharing) | X | X | 0.526275 | 96 msec | 2.8 Mb |
| 2 | PC | TFLite(Python, XNNPACK) | [deeplabv3_mnv2_dm05_pascal_trainval_dynamic_int.tflite](https://drive.google.com/file/d/1MkHYyDcX_AVUX1CrrpCOga5xICNDvAaA/view?usp=share_link)| int | X | 0.525301 | 75 msec | 964.2 Kb |
| 3 | PC | TFLite(C++, XNNPACK) | [deeplabv3_mnv2_dm05_pascal_trainval_dynamic_fp32.tflite](https://drive.google.com/file/d/1uy17pDW-JLcMXLDpGACBKmKbrstK0oYx/view?usp=sharing) | X | X | 0.585301 | 92 msec | 2.8 Mb |
| 4 | PC | TFLite(C++, XNNPACK) | [deeplabv3_mnv2_dm05_pascal_trainval_dynamic_int.tflite](https://drive.google.com/file/d/1MkHYyDcX_AVUX1CrrpCOga5xICNDvAaA/view?usp=share_link) | int | X | 0.574437 | 71 msec | 964.2 Kb |
| 5 | RPI3 | TFLite(C++, XNNPACK) | [deeplabv3_mnv2_dm05_pascal_trainval_dynamic_fp32.tflite](https://drive.google.com/file/d/1uy17pDW-JLcMXLDpGACBKmKbrstK0oYx/view?usp=sharing) | X | X | 0.585301 | 3176 msec | 2.8 Mb |
| 6 | RPI3 | TFLite(C++, XNNPACK) | [deeplabv3_mnv2_dm05_pascal_trainval_dynamic_int.tflite](https://drive.google.com/file/d/1LZp6or_o3DpbtR-4eYp8LDdPQuxDHjxt/view?usp=share_link) | int | X | 0.574511 | 2278 msec | 964.2 Kb |
| 7 | RB5 | SNPE (CPU) | [deeplabv3_mnv2_pascal_train_aug_2018_01_29_opt.dlc](https://drive.google.com/file/d/1fYG_DQ8sIChEb_BTagaLEWDyDbzSc5i6/view?usp=share_link) | X | X | 0.585294 | 413 msec | 2.9 Mb |
| 8 | RB5 | SNPE (GPU) | [deeplabv3_mnv2_pascal_train_aug_2018_01_29_opt.dlc](https://drive.google.com/file/d/1fYG_DQ8sIChEb_BTagaLEWDyDbzSc5i6/view?usp=share_link) | X | X | 0.585294 | 383 msec | 2.9 Mb |
