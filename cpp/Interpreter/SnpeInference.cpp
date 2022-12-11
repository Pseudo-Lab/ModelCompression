#include "SnpeInference.h"

#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"

#ifdef BOARD
static const std::string RUNTIME_TYPE = "GPU";
#else
static const std::string RUNTIME_TYPE = "CPU";
#endif


// TODO: set input dim. automatically
// TODO: store output layer names

CSnpeInference::CSnpeInference(const std::string& strWeightFilePath, const std::string& strConfigFilePath)
        : IInterpreter(strWeightFilePath, strConfigFilePath)
{
    Init(RUNTIME_TYPE, "ITENSOR");
}
//*
CSnpeInference::~CSnpeInference()
{
    if (m_pSnpe != nullptr)
    {
        m_pSnpe.release();   // TODO check diff between reset() and release()
        m_pContainer.release();
    }
}//*/

bool CSnpeInference::SetInputShape(int inWidth, int inHeight, int inChannels)
{
    m_inWidth =inWidth;
    m_inHeight =inHeight;
    m_inChannels =inChannels;
}

// TODO: merge Init() and LoadModel()
bool CSnpeInference::SetDelegate(DELEGATE _delegate)
{
    if(m_isLoadModel)
    {
        std::cout << "Please set delegate before load model!\n";
        return false;
    }

    return true;
}

bool CSnpeInference::Init(const std::string& strRuntimeType, const std::string& strBufferType)
{
    // Step 1. Set Runtime type
    m_runtime = zdl::DlSystem::Runtime_t::CPU;

    m_isRuntimeSpecified = true;
    if (strRuntimeType.compare("GPU") == 0)
    {
        m_runtime = zdl::DlSystem::Runtime_t::GPU;
    }
    else if (strRuntimeType.compare("AIP") == 0)
    {
        m_runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
    }
    else if (strRuntimeType.compare("DSP") == 0)
    {
        m_runtime = zdl::DlSystem::Runtime_t::DSP;
    }
    else
    {
        std::cout << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;
        m_isRuntimeSpecified = false;
    }

    // If quantized model -> USE DSP/AIP runtime
    if(m_isStaticQuantization)
    {
        m_runtime = zdl::DlSystem::Runtime_t::DSP;
    }

    // Check if given buffer type is valid
    m_bitWidth = 0;
    m_bufferType = BUFF_DATATYPE::ITENSOR; // defualt
    if (strBufferType.compare("USERBUFFER_FLOAT") == 0)
    {
        m_bufferType = BUFF_DATATYPE::USERBUFFER_FLOAT;
    }
    else if (strBufferType.compare("USERBUFFER_TF8") == 0)
    {
        m_bufferType = BUFF_DATATYPE::USERBUFFER_TF8;
        m_bitWidth = 8;
    }
    else if (strBufferType.compare("USERBUFFER_TF16") == 0)
    {
        m_bufferType = BUFF_DATATYPE::USERBUFFER_TF16;
        m_bitWidth = 16;
    }
    else
    {
        std::cout << "Buffer type is not valid. Defaulting to the ITENSOR setting" << std::endl;
    }

    // TODO
    //Check if given user buffer source type is valid
    m_userBufferSourceType = BUFF_TYPE::CPUBUFFER; // fix


    //Check if both runtimelist and runtime are passed in
    if (m_isRuntimeSpecified && m_runtimeList.empty() == false)
    {
        std::cout << "Invalid option cannot mix runtime order -l with runtime -r " << std::endl;
    }

    if (m_isRuntimeSpecified)
    {
        m_runtime = CheckRuntime(m_runtime, m_isStaticQuantization);
    }

    return true;
}

zdl::DlSystem::Runtime_t CSnpeInference::CheckRuntime(zdl::DlSystem::Runtime_t runtime, bool &staticQuantization)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number

    if((runtime != zdl::DlSystem::Runtime_t::DSP) && staticQuantization)
    {
        std::cerr << "ERROR: Cannot use static quantization with CPU/GPU runtimes. It is only designed for DSP/AIP runtimes.\n";
        std::cerr << "ERROR: Proceeding without static quantization on selected runtime.\n";
        staticQuantization = false;
    }

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
    {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    return runtime;
}

bool CSnpeInference::LoadModel()
{
    std::cout << "Load Model\n-> .dlc File: " << m_strWeightFilePath << "\n";

    // load dlc
    m_pContainer = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(m_strWeightFilePath.c_str()));

    if (m_pContainer == nullptr)
    {
        std::cerr << "Error while opening the container file." << std::endl;
        return false;
    }

    m_isUseUserSuppliedBuffers = (m_bufferType == BUFF_DATATYPE::USERBUFFER_FLOAT ||
                                  m_bufferType == BUFF_DATATYPE::USERBUFFER_TF8 ||
                                  m_bufferType == BUFF_DATATYPE::USERBUFFER_TF16);

    bool isUsingInitCaching = false;

    // SetBuilderOptions()
    zdl::SNPE::SNPEBuilder snpeBuilder(m_pContainer.get());

    if (m_runtimeList.empty())
    {
        m_runtimeList.add(m_runtime);
    }

    if (!m_vOutputLayerName.empty())
    {
        zdl::DlSystem::StringList strOutputLayerNames;

        for (auto& strLayerName : m_vOutputLayerName)
        {
            strOutputLayerNames.append(strLayerName.c_str());
        }

        // INFO: setInputDimensions() is only support PC environment.
        m_pSnpe = snpeBuilder.setOutputTensors(strOutputLayerNames)
                             .setRuntimeProcessorOrder(m_runtimeList)
                             .setUseUserSuppliedBuffers(m_isUseUserSuppliedBuffers)
                             .setPlatformConfig(m_platformConfig)
                             .setInitCacheMode(isUsingInitCaching)
                             .build();
    }
    else
    {
        m_pSnpe = snpeBuilder.setRuntimeProcessorOrder(m_runtimeList)
                             .setUseUserSuppliedBuffers(m_isUseUserSuppliedBuffers)
                             .setPlatformConfig(m_platformConfig)
                             .setInitCacheMode(isUsingInitCaching)
                             .build();
    }

    if (m_pSnpe == nullptr)
    {
        std::cerr << "Error while building SNPE object." << std::endl;
        return false;
    }

    // TODO: check routines for save and load Caching data
    if (isUsingInitCaching)
    {
        if (m_pContainer->save(m_strWeightFilePath))
        {
            std::cout << "Saved container into archive successfully" << std::endl;
        }
        else
        {
            std::cout << "Failed to save container into archive" << std::endl;
        }
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = m_pSnpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    size_t heightSize = tensorShape.getDimensions()[1];
    size_t widthSize = tensorShape.getDimensions()[2];
    size_t chSize = tensorShape.getDimensions()[3];

    std::cout << "-> input dimension for the container: "
              << batchSize << "x" << heightSize << "x" << widthSize << "x" << chSize << std::endl;

    m_isLoadModel = true;

    return true;
}

std::unique_ptr<zdl::DlSystem::ITensor> CSnpeInference::LoadInputTensor(const cv::Mat& srcImg)
{
    std::unique_ptr<zdl::DlSystem::ITensor> pInputTensor;

    if (m_pSnpe == nullptr)
    {
        std::cerr << "Please build SNPE first!!";

        return pInputTensor;
    }

    //Get input names and number
    const auto& inputTensorNamesRef = m_pSnpe->getInputTensorNames();
    if (!inputTensorNamesRef)
    {
        throw std::runtime_error("Error obtaining Input tensor names");
    }
    const auto& inputTensorNames = *inputTensorNamesRef;

    // Create an input tensor that is correctly sized to hold the input of the network.
    // Dimensions that have no fixed size will be represented with a value of 0.
    const auto& inputDims_opt = m_pSnpe->getInputDimensions(inputTensorNames.at(0));
    const auto& inputShape = *inputDims_opt;

    // Calculate the total number of elements that can be stored in the tensor
    // so that we can check that the input contains the expected number of elements.
    // With the input dimensions computed create a tensor to convey the input into the network.

    pInputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

    // Padding the input vector so as to make the size of the vector to equal to an integer multiple of the batch size
    zdl::DlSystem::TensorShape tensorShape = m_pSnpe->getInputDimensions();


    // Set input buffer from cv::Mat buffer
    cv::Mat targetImg;
    cv::resize(srcImg, targetImg, cv::Size(tensorShape.getDimensions()[2], tensorShape.getDimensions()[1]),
               cv::INTER_CUBIC);
    targetImg.convertTo(targetImg, CV_64F);

    double* pData = (double*) (targetImg.data);
    auto it = pInputTensor->begin();
    for (int k = 0; k < targetImg.rows; k++)
    {
        int rowStartIdx = k * targetImg.step1();
        for (int q = 0; q < targetImg.cols; q++)
        {
            int colStartIdx = q * 3;
            (*it) = (pData[rowStartIdx + colStartIdx + 2] - m_Mean[0]) * m_scale;
            ++it;
            (*it) = (pData[rowStartIdx + colStartIdx + 1] - m_Mean[1]) * m_scale;
            ++it;
            (*it) = (pData[rowStartIdx + colStartIdx + 0] - m_Mean[2]) * m_scale;
            ++it;
        }
    }

    return std::move(pInputTensor);
}

std::unordered_map<std::string, cv::Mat> CSnpeInference::Interpret(const cv::Mat& srcImg)
{

    std::unordered_map<std::string, cv::Mat> vResult;

    // m_bufferType : BUFF_DATATYPE::ITENSOR, Batch=1 only
    std::unique_ptr<zdl::DlSystem::ITensor> pInputTensor = LoadInputTensor(srcImg);

    if (!pInputTensor)
    {
        std::cerr << "Only surpport ITensor buffer";
        return vResult;
    }
    // A tensor map for SNPE execution outputs
    zdl::DlSystem::TensorMap outputTensorMap;

    // Execute the input tensor on the model with SNPE
    bool execStatus = m_pSnpe->execute(pInputTensor.get(), outputTensorMap);

    if (execStatus == true)
    {
        zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

        vResult.reserve(tensorNames.size());
        for (auto& name : tensorNames)
        {

            auto tensorPtr = outputTensorMap.getTensor(name);

            int size[6];
            cv::Mat curOutputLayer;
            int outputDim = outputTensorMap.size();

            if (outputDim == 1)
            {
                curOutputLayer.create(1,tensorPtr->getSize(), CV_32FC1);

                std::cout << "-> output tensor [" << name << "]: "
                          << 1 << "x" << tensorPtr->getSize() << "\n";
            }
            else
            {
                for (int k = 0; k < outputDim; k++)
                {
                    size[k] = tensorPtr->getShape()[k];
                }

                std::cout << "-> output tensor [" << name << "]: "
                          << size[0] << "x" << size[1] << "x" << size[2] << "x" << size[3] << "\n";

                curOutputLayer.create(outputDim, size, CV_32F);
            }
            float* pOutData = (float*) (curOutputLayer.data);
            int buffIdx = 0;
            for (auto it = tensorPtr->begin(); it < tensorPtr->end(); ++it, ++buffIdx)
            {
                pOutData[buffIdx] = (*it);
            }
            vResult.insert(std::make_pair(name, std::move(curOutputLayer)));
        }
    }
    else
    {
        std::cerr << "Error while executing the network." << std::endl;
    }
    return std::move(vResult);
}