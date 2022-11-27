#ifndef TFLITEINFERENCE_H
#define TFLITEINFERENCE_H

#include <unordered_map>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"

#include "IInterpreter.h"

class CTfLiteInterpreter : public IInterpreter
{
public:
    CTfLiteInterpreter(const std::string& strWeightFilePath, const std::string& strConfigFilePath);
    ~CTfLiteInterpreter() {}

    bool SetInputShape(int inWidth, int inHeight, int inChannels);
    bool SetDelegate(DELEGATE _delegate);
    bool LoadModel();

    std::unordered_map<std::string, cv::Mat> Interpret(const cv::Mat& srcImg);

protected:

    cv::Mat ConvertInputTensor(const cv::Mat& srcImg);
    cv::Size GetInputSpatialDim();

private:

    void ShowSummery();

    std::unique_ptr<tflite::FlatBufferModel> m_pModel;
    tflite::ops::builtin::BuiltinOpResolver m_resolver;
    std::unique_ptr<tflite::Interpreter> m_pInterpreter;

};
#endif //TFLITEINFERENCE_H

