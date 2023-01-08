#ifndef MIDAS_DEPTH_EST_H
#define MIDAS_DEPTH_EST_H

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>

//#include "../Interpreter/common/DetectionBox.h"
#include "../Interpreter/IInterpreter.h"

static constexpr int MIDAS_DEFAULT_IN_COLS = 384; // small 256
static constexpr int MIDAS_DEFAULT_IN_ROWS = 384; // small 256
static constexpr int MIDAS_DEFAULT_IN_CH = 3;

static constexpr int NUM_NYUDEPTHV2_TEST_SET = 654;

static cv::Scalar MIDAS_DEFAULT_MEAN = cv::Scalar(0.485, 0.456, 0.406);
static cv::Scalar MIDAS_DEFAULT_STD = cv::Scalar(0.229, 0.224, 0.225);


// reference model https://tfhub.dev/intel/midas/v2/2
class CMidasDepthEst
{
public:
    CMidasDepthEst(std::unique_ptr<IInterpreter> pInterpreter);
    
    ~CMidasDepthEst() {};

    cv::Mat Run(const cv::Mat& srcImg);
    void EvaluateNyuDepthV2(const std::string& strDbPath, const std::string& strSavePath, bool isInvertScale = false);
    cv::Mat DrawDepth(const cv::Mat& depthMat);

    void SetInvertedScale(bool isInvertScale);

private:

    cv::Mat LoadNyuDepthGtFromCsv(std::string& strGtPath);

    float GetAbsRel(const cv::Mat& predict, const cv::Mat& gt);
    float GetRmsError(const cv::Mat& predict, const cv::Mat& gt);
    float GetSiRmsError(const cv::Mat& predict, const cv::Mat& gt);

	template <typename T> cv::Mat GetLogMat(const cv::Mat& srcImg);

    std::vector <std::string> GetLabel(const std::string& strPath);

    bool m_isInvertScale;

    std::unique_ptr<IInterpreter> m_pInterpreter;
};

#endif // MIDAS_DEPTH_EST_H