#include "TfLiteInference.h"

CDnnInterpreter::CDnnInterpreter(int inWidth, int inHeight, int inChannels)
        : m_Mean(cv::Scalar(0,0,0)), m_scale(1.0), m_isReady(false)
{
    m_pModel = nullptr;
}

CDnnInterpreter::CDnnInterpreter(int inWidth, int inHeight, int inChannels, cv::Scalar mean, double scale)
        : m_Mean(mean), m_scale(scale), m_isReady(false)
{
    m_pModel = nullptr;
}

bool CDnnInterpreter::LoadModel(const std::string& strModelPath)
{
    m_isReady = true;

    // Step 1. Load Model
    m_pModel = tflite::FlatBufferModel::BuildFromFile(strModelPath.c_str());

    if(!m_pModel)
    {
        m_isReady = false;
        std::cerr << "Failed to mmap model!";
        return false;
    }

    // Step 2. Build the interpreter
    tflite::InterpreterBuilder(*m_pModel.get(), m_resolver)(&m_pInterpreter);

    if(!m_pInterpreter)
    {
        m_isReady = false;
        std::cerr << "Failed to construct interpreter";
        return false;
    }

    if(m_pInterpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to allocate tensor!";
        m_isReady = false;
        return false;
    }

    std::cout << "Load complete: " << strModelPath << "\n";
    ShowSummery();

    return m_isReady;
}

//* ToDo: Delete using "TfLiteGpuDelegateV2Delete( delegate_);"
void CDnnInterpreter::Release()
{
}

void CDnnInterpreter::ShowSummery()
{
    if(!m_isReady)
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

    std::cout << "-> Outputs: " << m_pInterpreter->outputs().size() << "\n";
    for(int iter =0; iter <m_pInterpreter->outputs().size(); iter++)
    {
        std::cout << "--> output(0) name: " << m_pInterpreter->GetOutputName(iter) << "\n";

        auto outputDim = m_pInterpreter->tensor(m_pInterpreter->outputs()[iter])->dims;

        std::cout << "--> output(" << iter << ") Shape = ["
                  << outputDim->data[0] << ", " << outputDim->data[1] << ", "
                  << outputDim->data[2]<< ", " << outputDim->data[3] << "]\n";
    }

}
std::unordered_map<std::string, cv::Mat> CDnnInterpreter::Interpret(const cv::Mat& srcImg)
{
    if(!m_isReady)
    {
        std::cerr << "Please Init. TFLite Interpreter!! ";
        return std::unordered_map<std::string, cv::Mat>();
    }

    // Step 1. Get [batch, ch, height, width] shape
    cv::Mat inputBlob = Resize(srcImg);

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

cv::Mat CDnnInterpreter::Resize(const cv::Mat& srcImg)
{
    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims->data;

    cv::Mat targetImg = srcImg.clone();

    cv::cvtColor(targetImg, targetImg, cv::COLOR_BGR2RGB);
    targetImg.convertTo(targetImg, CV_32F);
    targetImg = (targetImg - m_Mean) *m_scale;
    cv::resize(targetImg, targetImg, cv::Size(inputDim[2], inputDim[1]));

    return std::move(targetImg);
}

void CDnnInterpreter::SetInputMean(cv::Scalar value)
{
    m_Mean = value;
}

void CDnnInterpreter::SetInputScale(double scale)
{
    m_scale = scale;
}

cv::Size CDnnInterpreter::GetInputSpatialDim()
{
    if(!m_isReady)
    {
        std::cerr << "Please Init. TFLite Interpreter!! ";

        return cv::Size();
    }

    auto inputDim = m_pInterpreter->tensor(m_pInterpreter->inputs()[0])->dims;

    return cv::Size(inputDim->data[2], inputDim->data[1]);
}

