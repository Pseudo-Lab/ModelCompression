#ifndef DNNINFERENCE_H
#define DNNINFERENCE_H

#include <iostream>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "IInterpreter.h"


enum class LIB_TYPE
{
    NONE,
    DARKNET,
    TENSORFLOW,
    ONNX,
    TORCH,
    CAFFE
};

class CDnnInterpreter : public IInterpreter
{
public:
    CDnnInterpreter(const std::string& strWeightFilePath, const std::string& strConfigFilePath);

    ~CDnnInterpreter() {}

    bool SetInputShape(int inWidth, int inHeight, int inChannels);
    bool SetDelegate(DELEGATE _delegate);
    bool LoadModel();

    std::unordered_map<std::string, cv::Mat> Interpret(const cv::Mat& srcImg);

protected:
    cv::Mat ConvertInputTensor(const cv::Mat& srcImg);
    std::vector <cv::String> GetOutputsNames(const cv::dnn::Net& net);

private:
    cv::dnn::Net m_Net;

    LIB_TYPE GetLibType(const std::string& strConfigFilePath, const std::string& strWeightFilePath);

};

#endif //DNNINFERENCE_H

