#include <memory>

#include "opencv2/opencv.hpp"

#include "DeepLabv3/DeepLabv3.h"
#include "Classification/MnistEval.h"
#include "Classification/CifarEval.h"
#include "Classification/Classification.h"
//#include "CenterFace/CenterFace.h"
//#include "MiDaS/MidasDepthEst.h"
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
//void RunDeepLabv3();
//void RunCenterFace();

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
    /* MNIST
    std::string strModelPath ="../../../models/LeNet/pruned_model.tflite";
    std::string strTestImagePath ="../../datasets/MNIST/t10k-images-idx3-ubyte";;
    std::string strTestLabelPath ="../../datasets/MNIST/t10k-labels-idx1-ubyte";;
    //*/
    std::string strModelPath = "../../../models/MobileNet_pruned/model.tflite";
    std::string strTestImagePath = "../../datasets/CIFAR10/cifar-10-batches-bin/test_batch.bin";
    std::string strTestLabelPath = "../../datasets/CIFAR10/cifar-10-batches-bin/batches.meta.txt";

    float inputScale = 1.0 / 255.0; // MNIST CIFAR10
    //float inputScale = 1.0; // MobileNet pruned
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
        if (strcmp(argv[idx], "-modelPath") == 0)
        {
            strModelPath = argv[idx + 1];
        }
        if (strcmp(argv[idx], "-inputScale") == 0)
        {
            inputScale = atof(argv[idx + 1]);
        }
        if (strcmp(argv[idx], "-inputMean") == 0)
        {
            inputMean = cv::Scalar(atof(argv[idx + 1]),atof(argv[idx + 2]),atof(argv[idx + 3]));
        }
    }

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
            RunClassification(std::move(pInterpreter), strTestImagePath, strTestLabelPath);
            break;
        case OPERATION_MODE::SEG_PASCAL12_EVAL:
            RunClassification(std::move(pInterpreter), strTestImagePath, strTestLabelPath);
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
    /*
    std::string strModelPath = "../../../models/MobileNet_pruned/model.tflite";
    std::string strTestImagePath = "../../datasets/CIFAR10/cifar-10-batches-bin/test_batch.bin";
    std::string strTestLabelPath = "../../datasets/CIFAR10/cifar-10-batches-bin/batches.meta.txt";

    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CTfLiteInterpreter>(strModelPath, "");

    pInterpreter->SetInputShape(CIFAR10_DEFAULT_COLS, CIFAR10_DEFAULT_ROWS, CIFAR10_DEFAULT_CH);
    pInterpreter->SetInputMean(cv::Scalar(0.0,0.0,0.0));
    pInterpreter->SetInputScale(1.0/255.0);
    pInterpreter->SetInputOrderRgb(true);
    //*/
    std::unique_ptr<CCifar10Eval> pCifarEval = std::make_unique<CCifar10Eval>(std::move(pInterpreter));
    pCifarEval->EvaluateCifar10(strMnistImageBinaryPath, strMnistLabelBinaryPath);


}

void RunClassification(std::unique_ptr<IInterpreter> pInterpreter,
                       const std::string& strImagePath, const std::string& strLabelPath)
{
    /*
    std::string strModelPath = "../../../models/MobileNet/mobilenet_v1_1.0_224.tflite"; // scale 1/255
    //std::string strModelPath = "../../../models/MobileNet/mbv1_100_90_12b4_684.tflite"; // scale 1
    //std::string strTestImagePath = "../../datasets/test/grace_hopper.bmp";
    std::string strTestImagePath = "../../datasets/test/Ryanair.jpg";
    std::string strTestLabelPath = "../../datasets/test/labels.txt";

    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CTfLiteInterpreter>(strModelPath, "");

    //pInterpreter->SetInputShape(CLS_DEFAULT_COLS, CLS_DEFAULT_ROWS, CLS_DEFAULT_CH);
    pInterpreter->SetInputMean(cv::Scalar(0.0,0.0,0.0));
    pInterpreter->SetInputScale(1.0/255.0);
    //*/

    std::unique_ptr<CClassification> pClassiEval = std::make_unique<CClassification>(std::move(pInterpreter));

    pClassiEval->Run(strImagePath, strLabelPath);
}

void RunDeepLabv3(std::unique_ptr<IInterpreter> pInterpreter, const std::string& strImagePath)
{

    //static constexpr int INPUT_WIDTH = 513;
    //static constexpr int INPUT_HEIGHT = 513;
    //static constexpr int INPUT_CH = 3;
    //
    //static const cv::Scalar DEEPLAB_MEAN = cv::Scalar(127.5, 127.5, 127.5);
    //static constexpr double DEEPLAB_SCALE = 0.007843;

    //static const std::string TFLITE_MODEL_FILE = "../../../models/deeplabv3/tflite/deeplabv3_mnv2_dm05_pascal_trainval_fp32.tflite";
    //std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CTfLiteInterpreter>(TFLITE_MODEL_FILE, "");

    //pInterpreter->SetInputShape(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CH);
    //pInterpreter->SetInputMean(DEEPLAB_MEAN);
    //pInterpreter->SetInputScale(DEEPLAB_SCALE);


    //*
    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>(std::move(pInterpreter));

    cv::Mat srcImg = cv::imread("../../../res/deeplab1.png");

    auto t1 = std::chrono::steady_clock::now();
    cv::Mat maskMat = pDeepLabv3->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    //cv::imwrite("../res/result.png", maskMat);
}

void RunPascalVocEval(std::unique_ptr<IInterpreter> pInterpreter,
                      const std::string& strPascalVocPath)
{
    //std::string strVocPath = "../../datasets/VOCdevkit/VOC2012";

    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>(std::move(pInterpreter));
    pDeepLabv3->EvaluateVOC12Val(strPascalVocPath);
}


/*
static const std::string CENTERFACE_MODEL_FILE = "../res/centerface_640.dlc";
static constexpr int CENTERFACE_INPUT_WIDTH = 640;
static constexpr int CENTERFACE_INPUT_HEIGHT = 480;
static constexpr int CENTERFACE_INPUT_CH = 3;
static const cv::Scalar_<float> CENTERFACE_MEAN = cv::Scalar_<float>(0., 0., 0.);
static constexpr float CENTERFACE_SCALE = 1.;c constexpr int CENTERFACE_INPUT_HEIGHT = 480;
static constexpr int CENTERFACE_INPUT_CH = 3;
static const cv::Scalar_<float> CENTERFACE_MEAN = cv::Scalar_<float>(0., 0., 0.);
static constexpr float CENTERFACE_SCALE = 1.;


void RunCenterFace()
{
    std::unique_ptr<IInterpreter> pFaceDetInterpreter = std::make_unique<CSnpeInference>(CENTERFACE_MODEL_FILE, "");

    pFaceDetInterpreter->SetInputShape(CENTERFACE_INPUT_WIDTH, CENTERFACE_INPUT_HEIGHT, CENTERFACE_INPUT_CH);
    pFaceDetInterpreter->SetInputMean(CENTERFACE_MEAN);
    pFaceDetInterpreter->SetInputScale(CENTERFACE_SCALE);

    std::vector<cv::String> vstrOutputLayerName = {"537", "538", "539", "540"};
    pFaceDetInterpreter->SetOutputLayerName(vstrOutputLayerName);


    std::unique_ptr<CCenterFace> pFaceDetector = std::make_unique<CCenterFace>(std::move(pFaceDetInterpreter));

    cv::Mat srcImg = cv::imread("../res/000388.jpg");
    cv::Mat retImg = pFaceDetector->TestRun(srcImg);
    cv::imwrite("../res/retImg.jpg", retImg);

}
//*/

/*
static constexpr int MIDAS_DEFAULT_IN_COLS = 384; // small 256
static constexpr int MIDAS_DEFAULT_IN_ROWS = 384; // small 256
static constexpr int MIDAS_DEFAULT_IN_CH = 3;

static constexpr int NUM_NYUDEPTHV2_TEST_SET = 654;

static cv::Scalar MIDAS_DEFAULT_MEAN = cv::Scalar(0.485, 0.456, 0.406);
static cv::Scalar MIDAS_DEFAULT_STD = cv::Scalar(0.229, 0.224, 0.225);

void RunMiDaSDepth()
{

    std::string strModelPath = "../res/lite-model_midas_v2_1_small_1_lite_1.tflite";
    std::string strNyuDepthPath = "../res/DenseDepth";
    std::string strSavePath = "../res/result";
    bool isInvertScale = true;

    std::unique_ptr<IInterpreter> pInterpreter = std::make_unique<CTfLiteInterpreter>(strModelPath, "");

    pInterpreter->SetInputShape(MIDAS_DEFAULT_IN_COLS, MIDAS_DEFAULT_IN_ROWS, MIDAS_DEFAULT_IN_CH);
    pInterpreter->SetInputMean(MIDAS_DEFAULT_MEAN);
    pInterpreter->SetInputScale(MIDAS_DEFAULT_STD);

    std::unique_ptr<> pDepthEstimator = std::make_unique<CMidasDepthEst>(std::move(pInterpreter));

    pDepthEstimator->SetInvertedScale(isInvertScale);
    pDepthEstimator->Evaluate(strNyuDepthPath, strSavePath, isInvertScale);
    
}//*/