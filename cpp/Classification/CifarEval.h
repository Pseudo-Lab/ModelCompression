//
// Created by tysong on 22. 12. 5.
//

#ifndef _CIFAREVAL_H
#define _CIFAREVAL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

#include "../Interpreter/IInterpreter.h"

static constexpr int CIFAR10_DEFAULT_COLS= 32;
static constexpr int CIFAR10_DEFAULT_ROWS = 32;
static constexpr int CIFAR10_DEFAULT_CH = 3;
static constexpr int CIFAR10_DEFAULT_NUM_CLASS = 10;
static constexpr int CIFAR10_NUM_PIXEL = 1024;
static constexpr int CIFAR10_NUM_TEST_BATCH = 10000;

class CCifar10Eval
{
public:

    CCifar10Eval(std::unique_ptr<IInterpreter> pInterpreter);
    ~CCifar10Eval()
    {
    }
    float EvaluateCifar10(const std::string& strMnistImgPath, const std::string& strMnistLabelPath);



private:

    bool SetCifar10DataSet(const std::string& strCifarFilePath);
    int GetNextCifar10Img(cv::Mat& srcImg);
    void ReleaseFileStream();

    std::vector<std::string> GetLabel(const std::string& strPath);
    std::vector<int> GetSortedIndex(int numElement, float* pData);

    int m_numLabel;
    int m_width;
    int m_height;
    int m_ch;

    std::ifstream m_imgFileStream;

    std::vector<std::string> m_vLabels;

    unsigned int m_numEntry;
    unsigned int m_curEntry;

    bool m_isDataReady;
    std::unique_ptr<IInterpreter> m_pInterpreter;
};

#endif //_CIFAREVAL_H
