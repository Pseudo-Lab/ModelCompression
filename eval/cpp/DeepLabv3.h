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
#elif TFLite_RUNTIME
#include "TFLite/TfLiteInference.h"
#else
#include "OpenCV/DnnInference.h"
//*/
#endif

class CDeepLabv3 : public CDnnInterpreter
{
public:
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

