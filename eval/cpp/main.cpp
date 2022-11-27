#include <memory>

#include "opencv2/opencv.hpp"

#include "DeepLabv3.h"
#include "Interpreter/IInterpreter.h"

#if SNPE_RUNTIME
#include "Interpreter/SnpeInference.h"
#elif TFLite_RUNTIME
#include "Interpreter/TfLiteInference.h"
#else
#include "Interpreter/DnnInference.h"
#endif

static constexpr int INPUT_WIDTH = 513;
static constexpr int INPUT_HEIGHT = 513;
static constexpr int INPUT_CH = 3;


static const cv::Scalar DEEPLAB_MEAN = cv::Scalar(127.5, 127.5, 127.5);
static constexpr double DEEPLAB_SCALE = 0.007843;

static const std::string SNPE_MODEL_FILE = "../../../models/dlc/deeplabv3_mnv2_dm05_pascal_trainval.dlc";
static const std::string TFLITE_MODEL_FILE = "../../../models/tflite/deeplabv3_mnv2_dm05_pascal_trainval_int.tflite";
static const std::string TF_MODEL_FILE = "../../../models/tf/deeplabv3_mnv2_dm05_pascal_trainval_opt.pb";


int main(int argc, char** argv)
{

#if SNPE_RUNTIME
    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CSnpeInference>(SNPE_MODEL_FILE, "");
#elif TFLite_RUNTIME
    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CTfLiteInterpreter>(TFLITE_MODEL_FILE, "");
#else
    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CDnnInterpreter>(TF_MODEL_FILE, "");
#endif
    pInterpreter->SetInputShape(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CH);
    pInterpreter->SetInputMean(DEEPLAB_MEAN);
    pInterpreter->SetInputScale(DEEPLAB_SCALE);

    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>(std::move(pInterpreter));


    /* Test
    cv::Mat srcImg = cv::imread("../../../res/deeplab1.png");

    auto t1 = std::chrono::steady_clock::now();
    cv::Mat maskMat = pDeepLabv3->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    cv::imwrite("../res/result.png", maskMat);
    //*/

    // PASCAL 2012
#if BOARD
    std::string strVocPath = "datasets/VOCdevkit/VOC2012";
#else
    std::string strVocPath = "../../datasets/VOCdevkit/VOC2012";
#endif
    pDeepLabv3->EvaluateVOC12Val(strVocPath);

    return 0;
}
