//
// Created by taeyup.song on 22. 4. 28.
//

#include <numeric>
#include <algorithm>
#include <iostream>
#include <map>
#include "DeepLabv3.h"

#if SNPE_RUNTIME
static const std::string MODEL_FILE = "../../../models/dlc/deeplabv3.dlc";
#elif TFLite_RUNTIME
static const std::string MODEL_FILE = "../res/deeplabv3.tflite";
#else
static const std::string MODEL_FILE = "../res/deeplabv3.onnx";
#endif

static constexpr int INPUT_WIDTH = 513;
static constexpr int INPUT_HEIGHT = 513;
static constexpr int INPUT_CH = 3;

static constexpr double DEEPLAB_SCALE = 0.007843;

//*******************************************************************************************************

#if SNPE_RUNTIME
//*
CDeepLabv3::CDeepLabv3() :
        CDnnInterpreter(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CH, false, cv::Scalar(27.5, 127.5, 127.5), DEEPLAB_SCALE)
        , m_InferWidth(INPUT_WIDTH), m_InferHeight(INPUT_HEIGHT)
{
    Init(MODEL_FILE);
}//*/

CDeepLabv3::CDeepLabv3(const std::string& strModelPath) :
        CDnnInterpreter(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CH, false, cv::Scalar(27.5, 127.5, 127.5), DEEPLAB_SCALE)
        , m_InferWidth(INPUT_WIDTH), m_InferHeight(INPUT_HEIGHT)
{
    Init(strModelPath);
}
/*
#elif TFLite_RUNTIME
CCenterFace::CCenterFace() :
        CTfLiteInference(cv::Scalar(0, 0, 0), 1.0), m_InferWidth(CENTERFACE_INPUT_WIDTH),
        m_InferHeight(CENTERFACE_INPUT_HEIGHT)
{
    Init(CENTERFACE_MODEL_FILE);
}

CCenterFace::CCenterFace(const std::string& strModelPath) :
        CTfLiteInference(), m_InferWidth(CENTERFACE_INPUT_WIDTH), m_InferHeight(CENTERFACE_INPUT_HEIGHT)
{
    Init(strModelPath);
}
#else
CCenterFace::CCenterFace() :
        CDnnInference(CENTERFACE_INPUT_WIDTH, CENTERFACE_INPUT_HEIGHT, CENTERFACE_INPUT_CH, cv::Scalar(0, 0, 0), 1.0)
        , m_InferWidth(CENTERFACE_INPUT_WIDTH), m_InferHeight(CENTERFACE_INPUT_HEIGHT)
{
    Init(CENTERFACE_MODEL_FILE);
}

CDeepLabv3::CDeepLabv3(const std::string& strModelPath) :
        CDnnInference(CENTERFACE_INPUT_WIDTH, CENTERFACE_INPUT_HEIGHT, CENTERFACE_INPUT_CH)
        , m_InferWidth(CENTERFACE_INPUT_WIDTH), m_InferHeight(CENTERFACE_INPUT_HEIGHT)
{
    Init(strModelPath);
}
//*/
#endif

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



// TODO: convert INT
cv::Mat CDeepLabv3::Run(const cv::Mat& srcImg)
{
	cv::Mat outputMat;
    std::unordered_map<std::string, cv::Mat> vNetOutput = Interpret(srcImg);

    //*
    if (vNetOutput.empty())
    {
        std::cout << "end\n";
        return outputMat;
    }
    
    //outputMat = vNetOutput[];
    
    return std::move(outputMat);
}

