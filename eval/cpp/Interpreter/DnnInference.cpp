#include "DnnInference.h"


CDnnInterpreter::CDnnInterpreter(const std::string& strWeightFilePath, const std::string& strConfigFilePath)
                                : IInterpreter(strWeightFilePath, strConfigFilePath)
{
}

bool CDnnInterpreter::SetInputShape(int inWidth, int inHeight, int inChannels)
{
    m_inWidth =inWidth;
    m_inHeight =inHeight;
    m_inChannels =inChannels;
}

bool CDnnInterpreter::LoadModel()
{
    std::cout << "Load Model\n-> Config File: " << m_strConfigFilePath << "\n-> Weight File: " << m_strWeightFilePath
              << "\n";

    LIB_TYPE libType = GetLibType(m_strConfigFilePath, m_strWeightFilePath);

    switch (libType)
    {
        case LIB_TYPE::DARKNET:
            m_Net = cv::dnn::readNetFromDarknet(m_strConfigFilePath, m_strWeightFilePath);
            std::cout << "Load Darknet model\n";
            break;
        case LIB_TYPE::TENSORFLOW:
            if (m_strConfigFilePath.empty())
            {
                m_Net = cv::dnn::readNetFromTensorflow(m_strWeightFilePath);
            }
            else
            {
                m_Net = cv::dnn::readNetFromTensorflow(m_strWeightFilePath, m_strConfigFilePath);
            }
            std::cout << "Load TF model\n";
            break;
        case LIB_TYPE::ONNX:
            m_Net = cv::dnn::readNet(m_strWeightFilePath);
            std::cout << "Load ONNX model\n";
            break;
        case LIB_TYPE::TORCH:
            m_Net = cv::dnn::readNetFromTorch(m_strWeightFilePath);
            std::cout << "Load Torch model\n";
            break;
        case LIB_TYPE::CAFFE:
            m_Net = cv::dnn::readNetFromCaffe(m_strConfigFilePath, m_strWeightFilePath);
            std::cout << "Load Torch model\n";
            break;
        default:
            std::cerr << "Please Check config/model file path!\n";
            break;
    }

    if (m_Net.empty())
    {
        return false;
    }

    // TODO: SetDelegate();
    m_Net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_Net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    if(m_vOutputLayerName.empty())
    {
        m_vOutputLayerName = GetOutputsNames(m_Net);
    }

    m_isLoadModel = true;
    return true;
}

bool CDnnInterpreter::SetDelegate(DELEGATE _delegate)
{
    if(m_isLoadModel)
    {
        std::cout << "Please set delegate before load model!\n";
        return false;
    }

    if(_delegate != DELEGATE::CPU)
    {
        std::cout << "Only CPU delegate supported!\n";
        return false;
    }
    return true;
}

std::unordered_map<std::string, cv::Mat> CDnnInterpreter::Interpret(const cv::Mat& srcImg)
{
    if(!m_isLoadModel)
    {
        std::cerr << "Please load model first!!";
        return std::unordered_map<std::string, cv::Mat>();
    }

    if (srcImg.empty())
    {
        std::cerr << "Please Check input tensor\n";
        return std::unordered_map<std::string, cv::Mat>();
    }

    cv::Mat blob = ConvertInputTensor(srcImg);
    m_Net.setInput(blob);

    std::vector<cv::Mat> vNetOuts;
    m_Net.forward(vNetOuts, m_vOutputLayerName);

    std::unordered_map<std::string, cv::Mat> ormOutput;
    ormOutput.reserve(vNetOuts.size());
    for (unsigned int iter = 0; iter <vNetOuts.size(); iter++)
    {
        ormOutput.insert(std::make_pair(m_vOutputLayerName[iter], vNetOuts[iter].clone()));
        std::cout << "Output Dim.: ";
        for(int k =0; k < vNetOuts[iter].dims; ++k)
        {
            std::cout << vNetOuts[iter].size[k] << ",";
        }
        std::cout << "\n";
    }

    return std::move(ormOutput);
}

cv::Mat CDnnInterpreter::ConvertInputTensor(const cv::Mat& srcImg)
{
    cv::Mat blob = cv::dnn::blobFromImage(srcImg, m_scale, cv::Size(m_inWidth, m_inHeight), m_Mean, true, false);

    return std::move(blob);
}

std::vector <cv::String> CDnnInterpreter::GetOutputsNames(const cv::dnn::Net& net)
{
    static std::vector <cv::String> vstrNames;
    if (vstrNames.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> vOutLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector <cv::String> vstrLayersNames = net.getLayerNames();

        // Get the names of the output layers in names
        vstrNames.resize(vOutLayers.size());
        for (size_t i = 0; i < vOutLayers.size(); ++i)
        {
            vstrNames[i] = vstrLayersNames[vOutLayers[i] - 1];
        }
    }
    return std::move(vstrNames);
}

LIB_TYPE CDnnInterpreter::GetLibType(const std::string& strConfigFilePath, const std::string& strWeightFilePath)
{
    LIB_TYPE libType = LIB_TYPE::NONE;
    std::string strConfigExt = strConfigFilePath.substr(strConfigFilePath.find_last_of(".") + 1);
    std::string strWeightExt = strWeightFilePath.substr(strWeightFilePath.find_last_of(".") + 1);
    if (strConfigExt == "cfg" && strWeightExt == "weights")
    {
        libType = LIB_TYPE::DARKNET;
    }
    else if (strWeightExt == "pb")
    {
        libType = LIB_TYPE::TENSORFLOW;
    }
    else if (strWeightExt == "onnx")
    {
        libType = LIB_TYPE::ONNX;
    }
    else if (strWeightExt == "t7")
    {
        libType = LIB_TYPE::TORCH;
    }
    else if (strWeightExt == "caffemodel")
    {
        libType = LIB_TYPE::CAFFE;
    }

    return libType;
}