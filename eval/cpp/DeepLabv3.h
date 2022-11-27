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

#include "Interpreter/IInterpreter.h"

class CDeepLabv3
{
public:
    explicit CDeepLabv3(std::unique_ptr<IInterpreter> pInterpreter);
    explicit CDeepLabv3(std::unique_ptr<IInterpreter> pInterpreter,
                        const std::string& strWeightPath, const std::string& strConfigPath,
                        int _inWidth, int _inHeight, int _inCh, cv::Scalar mean, double scale);

    ~CDeepLabv3();

    cv::Mat Run(const cv::Mat& srcImg);
    void EvaluateVOC12Val(const std::string& strPath);


private:
    cv::Mat ColorizeSegmentationBCHW(const cv::Mat &score);
    cv::Mat ColorizeSegmentationHWC(const cv::Mat &score);

    std::unique_ptr<IInterpreter> m_pInterpreter;
};

#endif //DEEPLABV3_H

