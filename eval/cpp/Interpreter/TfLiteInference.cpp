#include "TfLiteInference.h"

CTfLiteInterpreter::CTfLiteInterpreter(const std::string& strWeightFilePath, const std::string& strConfigFilePath)
        : IInterpreter(strWeightFilePath, strConfigFilePath)
{
}

bool CTfLiteInterpreter::SetInputShape(int inWidth, int inHeight, int inChannels)
{
    std::cout << "TFLite interpretr not support dynamic input shape\n";

    return true;
}

bool CTfLiteInterpreter::SetDelegate(DELEGATE _delegate)
{
    if(m_isLoadModel)
    {
        std::cout << "Please set delegate before load model!\n";
        return false;
    }

    if(_delegate != DELEGATE::CPU)
    {
        std::cout << "Only CPU (XNNPACK) delegate supported!\n";
        return false;
    }
    return true;
}

bool CTfLiteInterpreter::LoadModel()
{
    m_isLoadModel = true;

    // Step 1. Load Model
    m_pModel = tflite::FlatBufferModel::BuildFromFile(m_strWeightFilePath.c_str());

    if(!m_pModel)
    {
        m_isLoadModel = false;
        std::cerr << "Failed to mmap model!";
        return false;
    }

    // Step 2. Build the interpreter
    tflite::InterpreterBuilder(*m_pModel.get(), m_resolver)(&m_pInterpreter);

    if(!m_pInterpreter)
    {
        m_isLoadModel = false;
        std::cerr << "Failed to construct interpreter";
        return false;
    }

    if(m_pInterpreter->AllocateTensors() != kTfLiteOk)
    {
        m_isLoadModel = false;
        std::cerr << "Failed to allocate tensor!";
        return false;
    }

    std::cout << "Load complete: " << m_strWeightFilePath << "\n";
    ShowSummery();

    return m_isLoadModel;
}

void CTfLiteInterpreter::ShowSummery()
{
    if(!m_isLoadModel)
    {
        return;
    }

    std::cout << "-> tensors size: " << m_pInterpreter->tensors_size() << "\n";
    std::cout << "-> nodes size: " << m_pInterpreter->nodes_size() << "\n";
    std::cout << "-> inputs: " << m_pInterpreter->inputs().size() << "\n";

    for(int iter =0; iter <m_pInterpreter->inputs().size(); iter++)
    {
        std::cout << "--> input(0) name: " << m_pInterpreter->GetInputName(iter) << "\n";

        auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[iter])->dims;
        std::cout << "--> input(" << iter << ") shape = ["
                  << inputDim->data[0] << ", " << inputDim->data[1] << ", "
                  << inputDim->data[2]<< ", " << inputDim->data[3] << "]\n";
    }

    // Set Input Shape Info.
    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims;
    m_inWidth = inputDim->data[2];
    m_inHeight =inputDim->data[1];
    m_inChannels =inputDim->data[3];

    std::cout << "-> Outputs: " << m_pInterpreter->outputs().size() << "\n";
    for(int iter =0; iter <m_pInterpreter->outputs().size(); iter++)
    {
        std::cout << "--> output(0) name: " << m_pInterpreter->GetOutputName(iter) << "\n";
        m_vOutputLayerName.push_back(m_pInterpreter->GetOutputName(iter));

        auto outputDim = m_pInterpreter->tensor(m_pInterpreter->outputs()[iter])->dims;

        std::cout << "--> output(" << iter << ") Shape = ["
                  << outputDim->data[0] << ", " << outputDim->data[1] << ", "
                  << outputDim->data[2]<< ", " << outputDim->data[3] << "]\n";
    }
}
std::unordered_map<std::string, cv::Mat> CTfLiteInterpreter::Interpret(const cv::Mat& srcImg)
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

    // Step 1. Get [batch, ch, height, width] shape
    cv::Mat inputBlob = ConvertInputTensor(srcImg);

    // Step 2. Set input data
    memcpy(m_pInterpreter->typed_input_tensor<float>(0), inputBlob.data, inputBlob.total() * inputBlob.elemSize());

    // Step 3. Run inference
    m_pInterpreter->Invoke();

    // Step 4. Store output tensor
    std::unordered_map<std::string, cv::Mat> vRetTensor;
    int numOutputLayer = m_pInterpreter->outputs().size();
    for(int iter =0; iter <numOutputLayer; iter++)
    {
        TfLiteTensor* tfOutputTensor = m_pInterpreter->tensor(m_pInterpreter->outputs()[iter]);

        auto outputDim = tfOutputTensor->dims->data;
        cv::Mat outputBlob(outputDim[1], outputDim[2], CV_32FC(outputDim[3]), tfOutputTensor->data.f);

        vRetTensor.insert(std::make_pair(tfOutputTensor->name, std::move(outputBlob)));
    }

    m_pInterpreter->ResetVariableTensors();

    return std::move(vRetTensor);
}

cv::Mat CTfLiteInterpreter::ConvertInputTensor(const cv::Mat& srcImg)
{
    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims->data;

    cv::Mat targetImg = srcImg.clone();

    cv::cvtColor(targetImg, targetImg, cv::COLOR_BGR2RGB);
    targetImg.convertTo(targetImg, CV_32F);
    targetImg = (targetImg - m_Mean) *m_scale;
    cv::resize(targetImg, targetImg, cv::Size(inputDim[2], inputDim[1]));

    return std::move(targetImg);
}

cv::Size CTfLiteInterpreter::GetInputSpatialDim()
{
    if(!m_isLoadModel)
    {
        std::cerr << "Please Init. TFLite Interpreter!! ";

        return cv::Size();
    }

    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims;

    return cv::Size(inputDim->data[2], inputDim->data[1]);
}

