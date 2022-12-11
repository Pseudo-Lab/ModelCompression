#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <chrono>

#include "DeepLabv3.h"

static constexpr int PASCAL_CLASSES = 21;
std::vector<cv::Vec3b> vVocColor = {{0,   0,   0},
                                    {0,   0,   128},
                                    {0,   128, 0},
                                    {0,   128, 128},
                                    {128, 0,   0},
                                    {128, 0,   128},
                                    {128, 128, 0},
                                    {128, 128, 128},
                                    {0,   0,   64},
                                    {0,   0,   192},
                                    {0,   192, 64},
                                    {0,   128, 192},
                                    {128, 0,   64},
                                    {128, 0,   192},
                                    {128, 128, 64},
                                    {128, 128, 192},
                                    {0,   64,  0},
                                    {0,   64,  128},
                                    {0,   192, 0},
                                    {0,   192, 128},
                                    {128, 64,  0}};

cv::Vec3b VOC_BOUND = {255, 255, 255};
cv::Vec3b ZERO_VEC = cv::Vec3b(0, 0, 0);

//*******************************************************************************************************


CDeepLabv3::CDeepLabv3(std::unique_ptr<IInterpreter> pInterpreter)
{
    m_pInterpreter = std::move(pInterpreter);
    m_pInterpreter->LoadModel();
}

CDeepLabv3::~CDeepLabv3()
{
}

cv::Mat CDeepLabv3::Run(const cv::Mat& srcImg)
{
    // TODO: output layer name 처리, unordered map?
    std::unordered_map<std::string, cv::Mat> vNetOutput = m_pInterpreter->Interpret(srcImg);

    cv::Mat outputMat;
    if(vNetOutput.empty())
    {
        std::cout << "end\n";
        return outputMat;
    }
#if SNPE_RUNTIME
    cv::Mat& outTensorMat = vNetOutput["ResizeBilinear_2:0"];
    outTensorMat = outTensorMat.reshape(PASCAL_CLASSES, 513); // TODO: remove magic number
    outputMat = ColorizeSegmentationHWC(outTensorMat);
#elif TFLite_RUNTIME
    outputMat = ColorizeSegmentationHWC(vNetOutput["ResizeBilinear_2"]);
#else
    outputMat = ColorizeSegmentationBCHW(vNetOutput["ResizeBilinear_2"]);
#endif
    return std::move(outputMat);
}

// from https://github.com/opencv/opencv/blob/master/samples/dnn/segmentation.cpp
cv::Mat CDeepLabv3::ColorizeSegmentationBCHW(const cv::Mat& score)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1);
    maxVal.setTo(cv::Scalar(0));
    for(int ch = 0; ch < chns; ch++)
    {
        for(int row = 0; row < rows; row++)
        {
            const float* ptrScore = score.ptr<float>(0, ch, row);
            uint8_t* ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float* ptrMaxVal = maxVal.ptr<float>(row);
            for(int col = 0; col < cols; col++)
            {
                if(ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar) ch;
                }
            }
        }
    }

    cv::Mat segm(rows, cols, CV_8UC3);
    for(int row = 0; row < rows; row++)
    {
        const uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b* ptrSegm = segm.ptr<cv::Vec3b>(row);
        for(int col = 0; col < cols; col++)
        {
            ptrSegm[col] = vVocColor[ptrMaxCl[col]];
        }
    }

    return std::move(segm);
}

cv::Mat CDeepLabv3::ColorizeSegmentationHWC(const cv::Mat& score)
{
    cv::Mat segm(score.rows, score.cols, CV_8UC3);

    float* scoreData = (float*) score.data;
    int step = score.step1();

    for(int row = 0; row < score.rows; row++)
    {
        cv::Vec3b* ptrSegm = segm.ptr<cv::Vec3b>(row);

        int rowIdx = row * step;
        for(int col = 0; col < score.cols; col++)
        {
            int colIdx = rowIdx + score.channels() * col;

            float maxVal = scoreData[colIdx];
            int maxIdx = 0;
            for(int ch = 0; ch < score.channels(); ch++)
            {
                if(scoreData[colIdx + ch] > maxVal)
                {
                    maxVal = scoreData[colIdx + ch];
                    maxIdx = ch;
                }
            }
            ptrSegm[col] = vVocColor[maxIdx];
        }
    }

    return std::move(segm);
}

void CDeepLabv3::EvaluateVOC12Val(const std::string& strPath)
{
    // TODO: check Model Init. flag
    std::string listPath = strPath + "/ImageSets/Segmentation/val.txt";
    std::ifstream inStm(listPath);

    std::string dbName;
    int imgCont = 0;
    float mIoU = 0.0;

    std::chrono::system_clock::time_point tempTime = std::chrono::system_clock::now();
    std::chrono::milliseconds totalProcTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            tempTime - tempTime);
    while(std::getline(inStm, dbName))
    {
        // Load GT
        std::string gtPath = strPath + "/SegmentationClass/" + dbName + ".png";
        cv::Mat gtMat = cv::imread(gtPath);

        // Load source img
        std::string srcPath = strPath + "/JPEGImages/" + dbName + ".jpg";
        cv::Mat srcImg = cv::imread(srcPath);

        std::cout << srcPath << "\n";

        float iou = 0.0f;
        if(!srcImg.empty())
        {
            imgCont++;

            // predict
            std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
            cv::Mat predMat = Run(srcImg);
            std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
            cv::resize(predMat, predMat, cv::Size(srcImg.cols, srcImg.rows), cv::INTER_NEAREST);
            std::chrono::milliseconds procTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime);
            totalProcTime += procTime;

            // Get IoU
            int intersectionValue = 0;
            int unionValue = 0;

            for(int row = 0; row < srcImg.rows; row++)
            {
                cv::Vec3b* pGtData = gtMat.ptr<cv::Vec3b>(row);
                cv::Vec3b* pPredData = predMat.ptr<cv::Vec3b>(row);
                for(int col = 0; col < srcImg.cols; col++)
                {
                    if((pGtData[col] != ZERO_VEC || pPredData[col] != ZERO_VEC))
                    {
                        if(pGtData[col] == pPredData[col])
                        {
                            intersectionValue++;
                        }
                        unionValue++;
                    }
                }
            }
            iou = (intersectionValue == 0 || unionValue == 0) ? 0.0f : (float) intersectionValue / (float) unionValue;
            mIoU += iou;

            std::cout << "=> IoU: " << iou << ", Proc. Time: " << procTime.count() << " [msec]\n";
        }
    }

    mIoU /= float(imgCont);
    totalProcTime /= float(imgCont);

    std::cout << "==========================================================\n";
    std::cout << "=> mIoU: " << mIoU << ", Average Proc. Time: " << totalProcTime.count() << " [msec]\n";
}