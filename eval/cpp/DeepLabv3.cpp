#include <numeric>
#include <algorithm>
#include <iostream>
#include <map>
#include <chrono>

#include "DeepLabv3.h"

#if SNPE_RUNTIME
static const std::string MODEL_FILE = "../../../models/dlc/deeplabv3_mnv2_dm05_pascal_trainval.dlc";
//static const std::string MODEL_FILE = "../../../models/dlc/deeplabv3_from_frozengraph.dlc";
#elif TFLite_RUNTIME
static const std::string MODEL_FILE = "../../../models/dlc/fp32.tflite"; // TODO
#else
static const std::string MODEL_FILE = "../../../models/tf/deeplabv3_mnv2_dm05_pascal_trainval_opt.pb";
#endif

static constexpr int INPUT_WIDTH = 513;
static constexpr int INPUT_HEIGHT = 513;
static constexpr int INPUT_CH = 3;

static constexpr double DEEPLAB_SCALE = 0.007843;

std::vector<cv::Vec3b> vVocColor = {{0,0,0},{0,0,128},{0,128,0},{0,128,128},
{128,0,0},{128,0,128},{128,128,0},{128,128,128},
{0,0,64},{0,0,192},{0,192,64},{0,128,192},
{128,0,64},{128,0,192},{128,128,64},{128,128,192},
{0,64,0},{0,64,128},{0,192,0},{0,192,128},{128,64,0}};

cv::Vec3b VOC_BOUND = {255,255,255};
cv::Vec3b ZERO_VEC = cv::Vec3b(0,0,0);

//*******************************************************************************************************

CDeepLabv3::CDeepLabv3() :
        CDnnInterpreter(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CH, cv::Scalar(127.5, 127.5, 127.5), DEEPLAB_SCALE)
        , m_InferWidth(INPUT_WIDTH), m_InferHeight(INPUT_HEIGHT)
{
    Init(MODEL_FILE);
}

CDeepLabv3::CDeepLabv3(const std::string& strModelPath) :
        CDnnInterpreter(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CH, cv::Scalar(127.5, 127.5, 127.5), DEEPLAB_SCALE)
        , m_InferWidth(INPUT_WIDTH), m_InferHeight(INPUT_HEIGHT)
{
    Init(strModelPath);
}

CDeepLabv3::CDeepLabv3(const std::string& strModelPath, int width, int height) :
        CDnnInterpreter(width, height, INPUT_CH, cv::Scalar(127.5, 127.5, 127.5), DEEPLAB_SCALE)
        , m_InferWidth(INPUT_WIDTH), m_InferHeight(INPUT_HEIGHT)
{
    Init(strModelPath);
}

CDeepLabv3::CDeepLabv3(int width, int height) :
        CDnnInterpreter(width, height, INPUT_CH, cv::Scalar(127.5, 127.5, 127.5), DEEPLAB_SCALE)
        , m_InferWidth(INPUT_WIDTH), m_InferHeight(INPUT_HEIGHT)
{
    Init(MODEL_FILE);
}

CDeepLabv3::~CDeepLabv3()
{
}

bool CDeepLabv3::Init(const std::string& strtModelPath)
{
    // Step 1. Load FaceNet Model
#if SNPE_RUNTIME
    return LoadModel(strtModelPath);
#elif TFLite_RUNTIME

    bool isSucess = LoadModel(strtModelPath);
    cv::Size spatialDim = GetInputSpatialDim();

    m_InferWidth = spatialDim.width;
    m_InferHeight = spatialDim.height;

    return isSucess;
#else
    return LoadModel("", strtModelPath);
#endif
}

cv::Mat CDeepLabv3::Run(const cv::Mat& srcImg)
{
#if SNPE_RUNTIME
    std::vector<int> vSize = {1,513,513,21};

    std::unordered_map<std::string, cv::Mat> vNetOutput = Interpret(srcImg, vSize);
#else
    std::unordered_map<std::string, cv::Mat> vNetOutput = Interpret(srcImg);
#endif
    cv::Mat outputMat;
    if (vNetOutput.empty())
    {
        std::cout << "end\n";
        return outputMat;
    }
#if SNPE_RUNTIME
    outputMat = ColorizeSegmentationBHWC(vNetOutput["ResizeBilinear_2:0"]);
#else
    outputMat = ColorizeSegmentationBCHW(vNetOutput["ResizeBilinear_2"]);
#endif
    return std::move(outputMat);
}

// from https://github.com/opencv/opencv/blob/master/samples/dnn/segmentation.cpp
cv::Mat CDeepLabv3::ColorizeSegmentationBCHW(const cv::Mat &score)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1);
    maxVal.setTo(cv::Scalar(0));
    for (int ch = 0; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    cv::Mat segm(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = vVocColor[ptrMaxCl[col]];
        }
    }

    return std::move(segm);
}

cv::Mat CDeepLabv3::ColorizeSegmentationBHWC(const cv::Mat &score)
{
    const int rows = score.size[1];
    const int cols = score.size[2];
    const int chns = score.size[3];

    cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    for (int row = 0; row < rows; row++)
    {
        uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
        for (int col = 0; col < cols; col++)
        {
            const float *ptrScore = score.ptr<float>(0, row, col);

            float maxVal = ptrScore[0];
            int maxIdx = 0;
            for (int ch = 0; ch < chns; ch++)
            {
                if (ptrScore[ch] > maxVal)
                {
                    maxVal = ptrScore[col];
                    maxIdx = ch;
                }
            }
            ptrMaxCl[col] = (uchar)maxIdx;
        }
    }

    cv::Mat segm(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = vVocColor[ptrMaxCl[col]];
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
    int imgCont =0;
    float mIoU = 0.0;

    std::chrono::system_clock::time_point tempTime = std::chrono::system_clock::now();
    std::chrono::milliseconds totalProcTime =std::chrono::duration_cast<std::chrono::milliseconds>(tempTime - tempTime);
    while(std::getline(inStm, dbName))
    {
        // Load GT
    	std::string gtPath = strPath + "/SegmentationClass/" + dbName + ".png";
    	cv::Mat gtMat = cv::imread(gtPath);

        // Load source img
        std::string srcPath = strPath + "/JPEGImages/" + dbName + ".jpg";
    	cv::Mat srcImg = cv::imread(srcPath);

    	std::cout << srcPath <<"\n";

        float iou = 0.0f;
        if(!srcImg.empty())
    	{
            imgCont++;

            // predict
            std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
            cv::Mat predMat = Run(srcImg);
            std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
            cv::resize(predMat, predMat, cv::Size(srcImg.cols, srcImg.rows), cv::INTER_NEAREST);
            std::chrono::milliseconds procTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            totalProcTime += procTime;

            // Get IoU
            int intersectionValue = 0;
            int unionValue = 0;

            for (int row = 0; row < srcImg.rows; row++)
            {
                cv::Vec3b *pGtData = gtMat.ptr<cv::Vec3b>(row);
                cv::Vec3b *pPredData = predMat.ptr<cv::Vec3b>(row);
                for (int col = 0; col < srcImg.cols; col++)
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
            iou = (intersectionValue == 0 || unionValue == 0) ? 0.0f : (float)intersectionValue/(float)unionValue;
            mIoU +=iou;

            std::cout << "=> IoU: " << iou << ", Proc. Time: "<< procTime.count() << " [msec]\n";
    	}
    }

    mIoU /= float(imgCont);
    totalProcTime /= float(imgCont);

    std::cout << "==========================================================\n";
    std::cout << "=> mIoU: " << mIoU << ", Average Proc. Time: "<< totalProcTime.count() << " [msec]\n";
}