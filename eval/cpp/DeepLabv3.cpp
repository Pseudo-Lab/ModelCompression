#include <numeric>
#include <algorithm>
#include <iostream>
#include <map>
#include "DeepLabv3.h"

#if SNPE_RUNTIME
static const std::string MODEL_FILE = "../../../models/dlc/deeplabv3_resnet50_coco-cd0a2569.dlc";
#elif TFLite_RUNTIME
static const std::string MODEL_FILE = "../res/deeplabv3.tflite"; // TODO
#else
static const std::string MODEL_FILE = "../../../models/tf/deeplabv3_mnv2_pascal_train_aug/optimized_graph.pb";
#endif

static constexpr int INPUT_WIDTH = 513;
static constexpr int INPUT_HEIGHT = 513;
static constexpr int INPUT_CH = 3;

static constexpr double DEEPLAB_SCALE = 0.007843;

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
    //return LoadModel(CENTERFACE_MODEL_FILE);
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
	
    std::unordered_map<std::string, cv::Mat> vNetOutput = Interpret(srcImg);

    cv::Mat outputMat;
    if (vNetOutput.empty())
    {
        std::cout << "end\n";
        return outputMat;
    }
       
    outputMat = ColorizeSegmentation(vNetOutput["ResizeBilinear_3"]);
    
    return std::move(outputMat);
}

// from https://github.com/opencv/opencv/blob/master/samples/dnn/segmentation.cpp
cv::Mat CDeepLabv3::ColorizeSegmentation(const cv::Mat &score)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];
    
    // Generate colors.
    std::vector<cv::Vec3b> colors;
    colors.push_back(cv::Vec3b());
    for (int i = 1; i < chns; ++i)
    {
        cv::Vec3b color;
        for (int j = 0; j < 3; ++j)
        {
            color[j] = (colors[i - 1][j] + rand() % 256) / 2;
        }
        colors.push_back(color);
    }
    
    cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++)
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
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }
    
    return std::move(segm);
}

