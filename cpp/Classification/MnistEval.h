//
// Created by tysong on 22. 12. 5.
//

#ifndef _MNISTEVAL_H
#define _MNISTEVAL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

#include "../Interpreter/IInterpreter.h"
#include "Classification.h"

static constexpr int LABEL_MAGIC = 2051;
static constexpr int IMG_MAGIC = 2049;

static constexpr int MNIST_DEFAULT_COLS = 28;
static constexpr int MNIST_DEFAULT_ROWS = 28;
static constexpr int MNIST_DEFAULT_CH = 1;
static constexpr int MNIST_DEFAULT_NUM_CLASS = 10;

class CMnistEval : public CClassification
{
public:

    CMnistEval(std::unique_ptr<IInterpreter> pInterpreter) : CClassification(std::move(pInterpreter)){}
    float EvaluateMnist(const std::string& strMnistImgPath, const std::string& strMnistLabelPath);

    virtual ~CMnistEval()
    {
        ReleaseFileStream();
    }

private:

    bool SetMnistDataSet(const std::string& strMnistImgPath, const std::string& strMnistLabelPath);
    cv::Mat GetNextMnistImg();
    int GetNextMnistLabel();
    void ReleaseFileStream();

    std::ifstream m_imgFileStream;
    std::ifstream m_labelFileStream;

    unsigned int m_numEntry;
    unsigned int m_curEntry;

    bool m_isDataReady;
};

#endif //_MNISTEVAL_H
