#include <memory>

#include "opencv2/opencv.hpp"

#include "DeepLabv3/DeepLabv3.h"
#include "Classification/MnistEval.h"
#include "Classification/CifarEval.h"
#include "Classification/Classification.h"
#include "Interpreter/IInterpreter.h"

#if SNPE_RUNTIME
#include "Interpreter/SnpeInference.h"
#endif
#if TFLite_RUNTIME
#include "Interpreter/TfLiteInference.h"
#endif
#include "Interpreter/DnnInference.h"

void RunMnistEval(std::unique_ptr<IInterpreter> pInterpreter,
                  const std::string& strMnistImageBinaryPath, const std::string& strMnistLabelBinaryPath);
void RunCifar10Eval(std::unique_ptr<IInterpreter> pInterpreter,
                    const std::string& strMnistImageBinaryPath, const std::string& strMnistLabelBinaryPath);
void RunClassification(std::unique_ptr<IInterpreter> pInterpreter,
                       const std::string& strImagePath, const std::string& strLabelPath);
void RunDeepLabv3(std::unique_ptr<IInterpreter> pInterpreter, const std::string& strImagePath);
void RunPascalVocEval(std::unique_ptr<IInterpreter> pInterpreter, const std::string& strPascalVocPath);

enum class OPERATION_MODE
{
    CLASSIFICATION = 0, MNIST_EVAL, CIFAR10_EVAL, SEGMENTATION, SEG_PASCAL12_EVAL
};

enum class INTERPRETER
{
    TFLite = 0, SNPE, OpenCV
};


int main(int argc, char** argv)
{
    OPERATION_MODE mode = OPERATION_MODE::CIFAR10_EVAL;
    INTERPRETER interpreter = INTERPRETER::TFLite;

    std::string strModelPath;
    std::string strTestImagePath;
    std::string strTestLabelPath;

    float inputScale = 1.0 / 255.0;
    cv::Scalar inputMean = cv::Scalar(0.0,0.0,0.0);

    for (int idx = 0; idx < argc - 1; idx++)
    {
        if (strcmp(argv[idx], "-mode") == 0)
        {
            if(strcmp(argv[idx + 1], "cls") == 0)
            {
                mode = OPERATION_MODE::CLASSIFICATION;
            }
            else if(strcmp(argv[idx + 1], "mnist") == 0)
            {
                mode = OPERATION_MODE::MNIST_EVAL;
            }
            else if(strcmp(argv[idx + 1], "cifar10") == 0)
            {
                mode = OPERATION_MODE::CIFAR10_EVAL;
            }
            else if(strcmp(argv[idx + 1], "seg") == 0)
            {
                mode = OPERATION_MODE::SEGMENTATION;
            }
            else if(strcmp(argv[idx + 1], "pascal") == 0)
            {
                mode = OPERATION_MODE::SEG_PASCAL12_EVAL;
            }
        }
        if (strcmp(argv[idx], "-inputScale") == 0)
        {
            inputScale = atof(argv[idx + 1]);
        }
        if (strcmp(argv[idx], "-inputMean") == 0)
        {
            inputMean = cv::Scalar(atof(argv[idx + 1]),atof(argv[idx + 2]),atof(argv[idx + 3]));
        }

        if (strcmp(argv[idx], "-modelPath") == 0)
        {
            strModelPath = argv[idx + 1];
        }
        if (strcmp(argv[idx], "-inputPath") == 0)
        {
            strTestImagePath = argv[idx + 1];
        }
        if (strcmp(argv[idx], "-labelPath") == 0)
        {
            strTestLabelPath = argv[idx + 1];
        }
    }

    std::cout << "Model Path: " << strModelPath << "\n";
    std::cout << "image(DB) Path: " << strTestImagePath << "\n";
    std::cout << "label Path: " << strTestLabelPath << "\n";

    // Load Model
    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CTfLiteInterpreter>(strModelPath, "");
    pInterpreter->SetInputMean(inputMean);
    pInterpreter->SetInputScale(inputScale);
    pInterpreter->SetInputOrderRgb(true);

    switch(mode)
    {
        case OPERATION_MODE::MNIST_EVAL:
            RunMnistEval(std::move(pInterpreter), strTestImagePath, strTestLabelPath);
            break;
        case OPERATION_MODE::CIFAR10_EVAL:
            RunCifar10Eval(std::move(pInterpreter), strTestImagePath, strTestLabelPath);
            break;
        case OPERATION_MODE::CLASSIFICATION:
            RunClassification(std::move(pInterpreter), strTestImagePath, strTestLabelPath);
            break;
        case OPERATION_MODE::SEGMENTATION:
            RunDeepLabv3(std::move(pInterpreter), strTestImagePath);
            break;
        case OPERATION_MODE::SEG_PASCAL12_EVAL:
            RunPascalVocEval(std::move(pInterpreter), strTestImagePath);
            break;
        default:
            std::cout << "Please set mode!!\n";
            break;
    }
    return 0;
}

void RunMnistEval(std::unique_ptr<IInterpreter> pInterpreter,
                  const std::string& strMnistImageBinaryPath, const std::string& strMnistLabelBinaryPath)
{
    std::unique_ptr<CMnistEval> pMnsitEval = std::make_unique<CMnistEval>(std::move(pInterpreter));
    pMnsitEval->EvaluateMnist(strMnistImageBinaryPath, strMnistLabelBinaryPath);
}

void RunCifar10Eval(std::unique_ptr<IInterpreter> pInterpreter,
                    const std::string& strMnistImageBinaryPath, const std::string& strMnistLabelBinaryPath)
{
    std::unique_ptr<CCifar10Eval> pCifarEval = std::make_unique<CCifar10Eval>(std::move(pInterpreter));
    pCifarEval->EvaluateCifar10(strMnistImageBinaryPath, strMnistLabelBinaryPath);
}

void RunClassification(std::unique_ptr<IInterpreter> pInterpreter,
                       const std::string& strImagePath, const std::string& strLabelPath)
{
    std::unique_ptr<CClassification> pClassiEval = std::make_unique<CClassification>(std::move(pInterpreter));
    pClassiEval->Run(strImagePath, strLabelPath);
}

void RunDeepLabv3(std::unique_ptr<IInterpreter> pInterpreter, const std::string& strImagePath)
{
    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>(std::move(pInterpreter));

    cv::Mat srcImg = cv::imread(strImagePath);

    auto t1 = std::chrono::steady_clock::now();
    cv::Mat maskMat = pDeepLabv3->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    cv::imwrite("result.png", maskMat);
}

void RunPascalVocEval(std::unique_ptr<IInterpreter> pInterpreter, const std::string& strPascalVocPath)
{
    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>(std::move(pInterpreter));
    pDeepLabv3->EvaluateVOC12Val(strPascalVocPath);
}