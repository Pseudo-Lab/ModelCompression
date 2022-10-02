//
// Created by taeyup.song on 22. 4. 28.
//

#ifndef DEEPLABV3_H
#define DEEPLABV3_H

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

#if SNPE_RUNTIME
#include "SNPE/SnpeInference.h"
/*
#elif TFLite_RUNTIME
#include "../dnn/TFLite/TfLiteInference.h"
#else
#include "../dnn/OpenCV/DnnInference.h"
//*/
#endif

#if SNPE_RUNTIME
class CDeepLabv3 : public CDnnInterpreter
/*
#elif TFLite_RUNTIME
class CDeepLabv3 : public CTfLiteInference
#else
class CDeepLabv3 : public CDnnInference
//*/
#endif
{
public:
/*
#if SNPE_RUNTIME
    //explicit CDeepLabv3();
    CDeepLabv3(const std::string& strModelPath);


#elif TFLite_RUNTIME
    explicit CDeepLabv3();
    explicit CDeepLabv3(const std::string& strModelPath);
#else
    explicit CDeepLabv3();
    explicit CDeepLabv3(const std::string& strModelPath);

#endif
//*/
	explicit CDeepLabv3();
	explicit CDeepLabv3(const std::string& strModelPath);
    ~CDeepLabv3();

    cv::Mat Run(const cv::Mat& srcImg);

private:

    bool Init(const std::string& strtModelPath);

    int m_InferWidth;
    int m_InferHeight;
};

#endif //DEEPLABV3_H

