//
// Created by tysong on 22. 12. 5.
//

#include "CifarEval.h"

#include <numeric>


using Time = std::chrono::steady_clock;

CCifar10Eval::CCifar10Eval(std::unique_ptr<IInterpreter> pInterpreter)
{
    m_pInterpreter = std::move(pInterpreter);
    m_pInterpreter->LoadModel();
}

float CCifar10Eval::EvaluateCifar10(const std::string& strCifarFilePath, const std::string& strCifarLabelPath)
{
    float accuracy = 0.0f;

    // Step 1. Preparing MNIST dataset
    if(!SetCifar10DataSet(strCifarFilePath))
    {
        std::cerr << "Failed to load dataset!!\n";
        return -1;
    }

    if(m_vLabels.empty())
    {
        m_vLabels = GetLabel(strCifarLabelPath);
    }

    Time::time_point startTime = Time::now();
    for(int iter =0; iter <m_numEntry; iter++)
    {
        // Step 2. Get Next Image and Label
        cv::Mat srcImg;
        int curLabel =GetNextCifar10Img(srcImg);

        if(curLabel <0)
        {
            break;
        }

        std::cout << "Load Img: " << srcImg.cols << "x" << srcImg.rows << "\n"; 


        // Step 3. Run inference
        std::unordered_map<std::string, cv::Mat> tfOutputTensor = m_pInterpreter->Interpret(srcImg);


        float* dataPtr = (float*)tfOutputTensor["StatefulPartitionedCall:0"].data;
        /*
        std::cout << "output Result = [";
        for(int iter =0; iter <10; iter++)
        {
            std::cout << dataPtr[iter];

            if(iter !=9)
                std::cout << ",";
            else
                std::cout << "], label = " <<curLabel<< "\n";
        }//*/
        
      
        // -> Step 3. get rank
        std::vector<int> vRank = GetSortedIndex(10, dataPtr);

        // -> Step 4. print classification Result
        std::cout << "Result: GT-> " << curLabel << ", Predict-> " << vRank[0] << "\n";
  
        if(vRank[0] == curLabel)
        {
            accuracy += 1.0;
        }//*/
    }
    Time::time_point curTime = Time::now();
    auto inferencTime = std::chrono::duration_cast<std::chrono::milliseconds>(curTime - startTime).count();
    std::cout << "Eval. time : " << inferencTime << "[msec]\n";
    std::cout << "Performance (Accuracy: " << accuracy/m_numEntry << "\n";

    return (accuracy/m_numEntry);
}

bool CCifar10Eval::SetCifar10DataSet(const std::string& strCifarFilePath)
{
    m_isDataReady = false;

    m_imgFileStream.open(strCifarFilePath, std::ios::in | std::ios::binary);
    if(!m_imgFileStream.good())
    {
        std::cout<<"Incorrect CIFAR data file  "<<std::endl;
        return m_isDataReady;
    }

    m_numEntry = CIFAR10_NUM_TEST_BATCH;
    m_curEntry =0;
    m_isDataReady = true;

    std::cout << "CIFAR10 Set is Ready!!\n";

    return m_isDataReady;
}
int CCifar10Eval::GetNextCifar10Img(cv::Mat& srcImg)
{
    if (!m_isDataReady) {
        return -1;
    }

    std::cout << "Load Image: " << m_curEntry << "/" << m_numEntry << "\n";

    if (!srcImg.empty())
    {
        srcImg.release();
    }

    unsigned char label;
    m_imgFileStream.read(reinterpret_cast<char*>(&label), 1);

    cv::Mat bgr[CIFAR10_DEFAULT_CH];
    for(int iter =CIFAR10_DEFAULT_CH-1; iter >= 0; iter--)
    {
        bgr[iter].create(CIFAR10_DEFAULT_COLS,CIFAR10_DEFAULT_ROWS,CV_8UC1);
        m_imgFileStream.read(reinterpret_cast<char*>(bgr[iter].data), CIFAR10_NUM_PIXEL);
    }
    cv::merge(bgr, CIFAR10_DEFAULT_CH, srcImg);

    //if(MNIST_DEFAULT_COLS != m_width || MNIST_DEFAULT_ROWS != m_height)
    //{
    //    cv::resize(srcImg, srcImg, cv::Size( CIFAR10_DEFAULT_COLS, CIFAR10_DEFAULT_ROWS ), cv::INTER_CUBIC);
    //}
    srcImg.convertTo(srcImg, CV_32FC3);

    m_curEntry++;

    return int(label);

}

void CCifar10Eval::ReleaseFileStream()
{
    if(m_isDataReady)
    {
        m_imgFileStream.close();

        m_numEntry =0;
        m_curEntry =0;

        m_isDataReady = false;
    }
}

std::vector<std::string> CCifar10Eval::GetLabel(const std::string& strPath)
{
    std::vector<std::string> vLabels;

    std::ifstream input(strPath);
    for(std::string line; std::getline( input, line );)
    {
        vLabels.push_back(line);
    }

    return std::move(vLabels);
}

std::vector<int> CCifar10Eval::GetSortedIndex(int numElement, float* pData)
{
    std::vector<int> vRank(numElement);
    std::iota(vRank.begin(),vRank.end(),0);
    std::sort(vRank.begin(),vRank.end(), [&](int i,int j){return pData[i]>pData[j];} );

    return std::move(vRank);
}