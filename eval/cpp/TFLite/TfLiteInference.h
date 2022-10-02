#ifndef TFLITEINFERENCE_H
#define TFLITEINFERENCE_H

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#include "opencv2/opencv.hpp"

#include <unordered_map>

class CDnnInterpreter
{
public:
    CDnnInterpreter();
    CDnnInterpreter(cv::Scalar mean, double scale);

    virtual ~CDnnInterpreter() {}

    virtual void Run() {}


protected:

    bool LoadModel(const std::string& strModelPath);
    virtual std::unordered_map<std::string, cv::Mat> Interpret(const cv::Mat& srcImg);
    cv::Mat Resize(const cv::Mat& srcImg);

    void SetInputMean(cv::Scalar value);
    void SetInputScale(double scale);

    cv::Size GetInputSpatialDim();

    void Release();

private:

    void ShowSummery();

    std::unique_ptr<tflite::FlatBufferModel> m_pModel;
    tflite::ops::builtin::BuiltinOpResolver m_resolver;
    std::unique_ptr<tflite::Interpreter> m_pInterpreter;

    TfLiteDelegate * m_pDelegate;

    cv::Scalar m_Mean;
    double m_scale;

    bool m_isReady;

};
#endif //TFLITEINFERENCE_H

