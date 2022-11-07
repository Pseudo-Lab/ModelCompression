#include "DnnInference.h"


CDnnInterpreter::CDnnInterpreter(int inWidth, int inHeight, int inChannels)
        : m_inWidth(inWidth), m_inHeight(inHeight), m_inChannels(inChannels),
          m_Mean(cv::Scalar()), m_scale(1. / 255)
{
}

CDnnInterpreter::CDnnInterpreter(int inWidth, int inHeight, int inChannels, cv::Scalar mean, double scale)
        : m_inWidth(inWidth), m_inHeight(inHeight), m_inChannels(inChannels),
          m_Mean(mean), m_scale(scale)
{
}

bool CDnnInterpreter::LoadModel(const std::string& strConfigFilePath, const std::string& strWeightFilePath,
                              const std::vector<cv::String>& vOutputLayerNames)
{

    std::cout << "Load Model\n-> Config File: " << strConfigFilePath << "\n-> Weight File: " << strWeightFilePath
              << "\n";

    LIB_TYPE libType = GetLibType(strConfigFilePath, strWeightFilePath);


    switch (libType)
    {
        case LIB_TYPE::DARKNET:
            m_Net = cv::dnn::readNetFromDarknet(strConfigFilePath, strWeightFilePath);
            std::cout << "Load Darknet model\n";
            break;
        case LIB_TYPE::TENSORFLOW:
            if (strConfigFilePath.empty())
            {
                m_Net = cv::dnn::readNetFromTensorflow(strWeightFilePath);
            }
            else
            {
                m_Net = cv::dnn::readNetFromTensorflow(strWeightFilePath, strConfigFilePath);
            }
            std::cout << "Load TF model\n";
            break;
        case LIB_TYPE::ONNX:
            m_Net = cv::dnn::readNet(strWeightFilePath);
            std::cout << "Load ONNX model\n";
            break;
        case LIB_TYPE::TORCH:
            m_Net = cv::dnn::readNetFromTorch(strWeightFilePath);
            std::cout << "Load Torch model\n";
            break;
        case LIB_TYPE::CAFFE:
            m_Net = cv::dnn::readNetFromCaffe(strConfigFilePath, strWeightFilePath);
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

    m_Net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_Net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    if(vOutputLayerNames.empty())
    {
        m_vOutputLayerName = GetOutputsNames(m_Net);
    }
    else
    {
        std::copy(vOutputLayerNames.begin(), vOutputLayerNames.end(), m_vOutputLayerName.begin());
    }
    return true;
}

std::unordered_map<std::string, cv::Mat> CDnnInterpreter::Interpret(const cv::Mat& srcImg)
{
    if (m_Net.empty() || srcImg.empty())
    {
        std::cerr << "Please Check model or input image\n";
        return std::unordered_map<std::string, cv::Mat>();
    }

    std::vector<cv::Mat> vNetOuts;

    cv::Mat blob = cv::dnn::blobFromImage(srcImg, m_scale, cv::Size(m_inWidth, m_inHeight), m_Mean, true, false);
    m_Net.setInput(blob);
    m_Net.forward(vNetOuts, m_vOutputLayerName);

    //
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
    }//*/

    return std::move(ormOutput);
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

void CDnnInterpreter::SetInputMean(cv::Scalar value)
{
    m_Mean = value;
}

void CDnnInterpreter::SetInputScale(double scale)
{
    m_scale = scale;
}
