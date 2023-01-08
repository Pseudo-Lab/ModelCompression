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

    bool isSaveOutLayerName = false;
    if(m_vOutputLayerName.empty())
    {
        isSaveOutLayerName = true;
    }

    std::cout << "-> tensors size: " << m_pInterpreter->tensors_size() << "\n";
    std::cout << "-> nodes size: " << m_pInterpreter->nodes_size() << "\n";
    std::cout << "-> inputs: " << m_pInterpreter->inputs().size() << "\n";

    for(int iter =0; iter <m_pInterpreter->inputs().size(); iter++)
    {
        std::cout << "--> input(0) name: " << m_pInterpreter->GetInputName(iter) << "\n";

        auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[iter])->dims;

        std::cout << "--> input(" << iter << ") shape = [";
        for(int q =0; q <inputDim->size; q++)
        {
            std::cout << inputDim->data[q];
            if(q == inputDim->size-1)
            {
                std::cout << "]\n";
            }
            else
            {
                std::cout << ", ";
            }
        }
    }

    // Set Input Shape Info.
    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims;

    m_inWidth = inputDim->data[2];
    m_inHeight = inputDim->data[1];
    m_inChannels = (inputDim->size ==4) ? inputDim->data[3] : 1;

    std::cout << "-> Outputs: " << m_pInterpreter->outputs().size() << "\n";
    for(int iter =0; iter <m_pInterpreter->outputs().size(); iter++)
    {
        std::cout << "--> output(" <<iter <<") name: " << m_pInterpreter->GetOutputName(iter) << "\n";
        if(isSaveOutLayerName)
        {
            m_vOutputLayerName.push_back(m_pInterpreter->GetOutputName(iter));
        }

        auto outputDim = m_pInterpreter->tensor(m_pInterpreter->outputs()[iter])->dims;

        std::cout << "--> output(" << iter << ") Shape = [";
        for(int q =0; q <outputDim->size; q++)
        {
            std::cout << outputDim->data[q];
            if(q == outputDim->size-1)
            {
                std::cout << "]\n";
            }
            else
            {
                std::cout << ", ";
            }
        }
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

        cv::Mat outputBlob;
        if(tfOutputTensor->dims->size <3)
        {
            outputBlob = cv::Mat(1, std::max(outputDim[1], outputDim[2]), CV_32F, tfOutputTensor->data.f);
        }
        else
        {
            int outputCh = (outputDim[3] ==0) ? 1 : outputDim[3];
            outputBlob = cv::Mat(outputDim[1], outputDim[2], CV_32FC(outputCh), tfOutputTensor->data.f);
        }
        vRetTensor.insert(std::make_pair(tfOutputTensor->name, std::move(outputBlob)));
    }
    m_pInterpreter->ResetVariableTensors();

    return std::move(vRetTensor);
}

cv::Mat CTfLiteInterpreter::ConvertInputTensor(const cv::Mat& srcImg)
{
    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims->data;

    cv::Mat targetImg = srcImg.clone();

    if(m_inChannels == 1)
    {
        if(srcImg.channels() ==3)
        {
            cv::cvtColor(targetImg, targetImg, cv::COLOR_BGR2GRAY);
        }
        targetImg.convertTo(targetImg, CV_32F, m_scale);
    }
    else if(m_inChannels == 3)
    {
        if(m_isOrderRgb)
        {
            cv::cvtColor(targetImg, targetImg, cv::COLOR_BGR2RGB);
        }
        targetImg.convertTo(targetImg, CV_32F);
        // DIM
        targetImg = (targetImg - m_Mean) * m_scale;
    }
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

