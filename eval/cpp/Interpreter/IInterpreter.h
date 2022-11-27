//
// Created by tysong on 22. 11. 26.
//

#ifndef _IINTERPRETER_H
#define _IINTERPRETER_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>

enum class DELEGATE
{
    CPU = 0, GPU, DSP, AIP
};

class IInterpreter {
public:

    IInterpreter(const std::string& strWeightFilePath, const std::string& strConfigFilePath)
            : m_strWeightFilePath(strWeightFilePath), m_strConfigFilePath(strConfigFilePath),
              m_inWidth(0), m_inHeight(0), m_inChannels(0),
              m_Mean(cv::Scalar()), m_scale(1),
              m_isLoadModel(false),
              m_delegate(DELEGATE::CPU)
    {}

    virtual ~IInterpreter() {};

    virtual bool SetInputShape(int inWidth, int inHeight, int inChannels) = 0;
    virtual bool SetDelegate(DELEGATE _delegate) =0;
    virtual bool LoadModel() =0;

    virtual std::unordered_map<std::string, cv::Mat> Interpret(const cv::Mat& srcImg) =0;

    void SetInputMean(cv::Scalar value);
    void SetInputScale(double scale);
    void SetOutputLayerName(const std::vector<cv::String>& vStrLayerName);

    std::vector<cv::String> m_vOutputLayerName;

protected:

    std::string m_strWeightFilePath;
    std::string m_strConfigFilePath;

    int m_inWidth;
    int m_inHeight;
    int m_inChannels;

    double m_scale;
    cv::Scalar m_Mean;

    bool m_isLoadModel;

    DELEGATE m_delegate;
};

#endif //_IINTERPRETER_H
