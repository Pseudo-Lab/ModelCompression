//
// Created by tysong on 22. 12. 7.
//

#ifndef EVALUATION_CLASSIFICATION_H
#define EVALUATION_CLASSIFICATION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

#include "../Interpreter/IInterpreter.h"


static constexpr int CLS_DEFAULT_COLS = 224;
static constexpr int CLS_DEFAULT_ROWS = 224;
static constexpr int CLS_DEFAULT_CH = 3;
static constexpr int CLS_DEFAULT_NUM_CLASS = 1000;

class CClassification
{
public:
    CClassification(std::unique_ptr<IInterpreter> pInterpreter);
    ~CClassification()
    {
    }
    int Run(const std::string& strMnistImgPath, const std::string& strMnistLabelPath);



protected:

    std::vector<std::string> GetLabel(const std::string& strPath);
    cv::Mat LoadImage(const std::string& strPath);
    std::vector<int> GetSortedIndex(int numElement, float* pData);

    int m_numLabel;
    std::vector<std::string> m_vLabels;

    std::unique_ptr<IInterpreter> m_pInterpreter;

    std::vector<std::string> m_strOutputName;
};

#endif //EVALUATION_CLASSIFICATION_H
