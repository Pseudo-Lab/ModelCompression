//
// Created by tysong on 22. 12. 7.
//

#include "Classification.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

using Time = std::chrono::steady_clock;

CClassification::CClassification(std::unique_ptr<IInterpreter> pInterpreter)
{
    m_pInterpreter = std::move(pInterpreter);
    m_pInterpreter->LoadModel();

    m_strOutputName = m_pInterpreter->GetOutputLayerName();
}

std::vector<std::string> CClassification::GetLabel(const std::string& strPath)
{
    std::vector<std::string> vLabels;

    std::ifstream input(strPath);
    for(std::string line; std::getline( input, line );)
    {
        vLabels.push_back(line);
    }

    return std::move(vLabels);
}

std::vector<int> CClassification::GetSortedIndex(int numElement, float* pData)
{
    std::vector<int> vRank(numElement);
    std::iota(vRank.begin(),vRank.end(),0);
    std::sort(vRank.begin(),vRank.end(), [&](int i,int j){return pData[i]>pData[j];} );

    return std::move(vRank);
}

int CClassification::Run(const std::string& strImgPath, const std::string& strLabelPath)
{

    if(m_vLabels.empty())
    {
        m_vLabels = GetLabel(strLabelPath);
    }

    // Step 1. Load Image
    //cv::Mat srcImg = LoadImage(strImgPath);

    cv::Mat srcImg = cv::imread(strImgPath);

    // Step 2. Run inference
    Time::time_point startTime = Time::now();
    std::unordered_map<std::string, cv::Mat> tfOutputTensor  = m_pInterpreter->Interpret(srcImg);
    Time::time_point curTime = Time::now();
    auto inferencTime = std::chrono::duration_cast<std::chrono::milliseconds>(curTime - startTime).count();

    //auto pOutputData = tfOutputTensor->data.f;
    float* pOutputData = (float*)tfOutputTensor[m_strOutputName[0]].data;
    //float* pOutputData = (float*)tfOutputTensor["resnet_model/final_dense_1"].data;


    // -> Step 3. get rank
    std::vector<int> vRank = GetSortedIndex(CLS_DEFAULT_NUM_CLASS, pOutputData);

    // -> Step 4. print top-5 classification Result
    for(int iter=0; iter<5; iter++)
    {
        std::cout << "Top-" << iter << ": (idx= " << iter
                  << ", label= " << m_vLabels[vRank[iter]]
                  << "), prob.: " << pOutputData[vRank[iter]] << "\n";
    }

    std::cout << "Eval. time (only inference): " << inferencTime << "[msec]\n";

    return vRank[0];
}
