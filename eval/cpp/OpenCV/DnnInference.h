#ifndef DNNINFERENCE_H
#define DNNINFERENCE_H

#include <iostream>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


enum class LIB_TYPE
{
    NONE,
    DARKNET,
    TENSORFLOW,
    ONNX,
    TORCH,
    CAFFE
};

class CDnnInterpreter
{
public:
    CDnnInterpreter(int inWidth, int inHeight, int inChannels);
    CDnnInterpreter(int inWidth, int inHeight, int inChannels, cv::Scalar mean, double scale);

    virtual ~CDnnInterpreter() {}

    virtual void Run() {}
    bool LoadModel(const std::string& strConfigFilePath, const std::string& strWeightFilePath,
                   const std::vector<cv::String>& vOutputLayerNames=std::vector<cv::String>());

protected:
    virtual std::unordered_map<std::string, cv::Mat> Interpret(const cv::Mat& srcImg);

    std::vector <cv::String> GetOutputsNames(const cv::dnn::Net& net);
    void SetInputMean(cv::Scalar value);
    void SetInputScale(double scale);


private:
    cv::dnn::Net m_Net;

    int m_inWidth;
    int m_inHeight;
    int m_inChannels;

    std::string m_strInputNodeName;

    cv::Scalar m_Mean;
    double m_scale;

    std::vector <cv::String> m_vOutputLayerName;

    LIB_TYPE GetLibType(const std::string& strConfigFilePath, const std::string& strWeightFilePath);

};

#endif //DNNINFERENCE_H

