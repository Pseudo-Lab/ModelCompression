//
// Created by tysong on 22. 12. 5.
//

#include "MnistEval.h"

#include <numeric>



using Time = std::chrono::steady_clock;

/*
CMnistEval::CMnistEval(std::unique_ptr<IInterpreter> pInterpreter)
{
    m_pInterpreter = std::move(pInterpreter);
    m_pInterpreter->LoadModel();
}//*/

float CMnistEval::EvaluateMnist(const std::string& strMnistImgPath, const std::string& strMnistLabelPath)
{
    float accuracy = 0.0f;

    // Step 1. Preparing MNIST dataset
    if(!SetMnistDataSet(strMnistImgPath, strMnistLabelPath))
    {
        return -1;
    }

    Time::time_point startTime = Time::now();
    for(int iter =0; iter <m_numEntry; iter++)
    {
        // Step 2. Get Next Image and Label
        cv::Mat srcImg =GetNextMnistImg();
        int curLabel = GetNextMnistLabel();

        if(srcImg.empty())
        {
            break;
        }

        // Step 3. Run inference

        std::unordered_map<std::string, cv::Mat> tfOutputTensor = m_pInterpreter->Interpret(srcImg);
        // -> Step 3. get rank
        std::vector<int> vRank = GetSortedIndex(MNIST_DEFAULT_NUM_CLASS, (float*)tfOutputTensor["StatefulPartitionedCall:0"].data);

        // -> Step 4. print classification Result
        std::cout << "Result: GT-> " << curLabel << ", Predict-> " << vRank[0] << "\n";
        if(vRank[0] == curLabel)
        {
            accuracy += 1.0;
        }
    }
    Time::time_point curTime = Time::now();
    auto inferencTime = std::chrono::duration_cast<std::chrono::milliseconds>(curTime - startTime).count();
    std::cout << "Eval. time (only inference): " << inferencTime << "[msec]\n";
    std::cout << "Performance (Accuracy: " << accuracy/m_numEntry << ")\n";

    return (accuracy/m_numEntry);
}

// Reference : https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
bool CMnistEval::SetMnistDataSet(const std::string& strMnistImgPath, const std::string& strMnistLabelPath)
{
    m_isDataReady = false;
    auto SwapEndian = [](unsigned int val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    };

    m_imgFileStream.open(strMnistImgPath, std::ios::in | std::ios::binary);
    m_labelFileStream.open(strMnistLabelPath, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    int magic;
    unsigned int num_items;
    unsigned int num_labels;
    unsigned int rows;
    unsigned int cols;

    m_imgFileStream.read(reinterpret_cast<char*>(&magic), 4);
    magic = SwapEndian(magic);
    if(magic != LABEL_MAGIC)
    {
        std::cout<<"Incorrect image file magic: "<<magic<<std::endl;
        return m_isDataReady;
    }

    m_labelFileStream.read(reinterpret_cast<char*>(&magic), 4);
    magic = SwapEndian(magic);
    if(magic != IMG_MAGIC)
    {
        std::cout<<"Incorrect label file magic: "<<magic<<std::endl;
        return m_isDataReady;
    }

    m_imgFileStream.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = SwapEndian(num_items);
    m_labelFileStream.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = SwapEndian(num_labels);

    if(num_items != num_labels)
    {
        std::cout<<"image file nums should equal to label num"<<std::endl;
        return m_isDataReady;
    }

    m_imgFileStream.read(reinterpret_cast<char*>(&rows), 4);
    rows = SwapEndian(rows);
    m_imgFileStream.read(reinterpret_cast<char*>(&cols), 4);
    cols = SwapEndian(cols);

    std::cout<<"image and label num is: "<<num_items<<std::endl;
    std::cout<<"image rows: "<<rows<<", cols: "<<cols<<std::endl;


    m_numEntry = num_items;
    m_curEntry =0;

    m_isDataReady = true;

    std::cout << "MNIST Set is Ready!!\n";

    return m_isDataReady;
}

cv::Mat CMnistEval::GetNextMnistImg()
{
    if(!m_isDataReady)
    {
        return cv::Mat();
    }

    cv::Mat srcImg(MNIST_DEFAULT_COLS, MNIST_DEFAULT_ROWS, CV_8UC1);
    unsigned char* srcData = (unsigned char*)srcImg.data;
    m_imgFileStream.read((char*)srcData, MNIST_DEFAULT_COLS * MNIST_DEFAULT_ROWS);
    m_curEntry++;

    std::cout << "Load Image: " << m_curEntry << "/" << m_numEntry << "\n";

    return std::move(srcImg);
}

int CMnistEval::GetNextMnistLabel()
{
    if(!m_isDataReady)
    {
        return -1;
    }
    char label;
    m_labelFileStream.read(&label, 1);

    return int(label);
}

void CMnistEval::ReleaseFileStream()
{
    if(m_isDataReady)
    {
        m_imgFileStream.close();
        m_labelFileStream.close();

        m_numEntry =0;
        m_curEntry =0;

        m_isDataReady = false;
    }
}
/*
std::vector<int> CMnistEval::GetSortedIndex(int numElement, float* pData)
{
    std::vector<int> vRank(numElement);
    std::iota(vRank.begin(),vRank.end(),0);
    std::sort(vRank.begin(),vRank.end(), [&](int i,int j){return pData[i]>pData[j];} );

    return std::move(vRank);
}//*/