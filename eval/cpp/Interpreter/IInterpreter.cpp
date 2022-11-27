#include "IInterpreter.h"

void IInterpreter::SetInputMean(cv::Scalar value)
{
    m_Mean = value;
}

void IInterpreter::SetInputScale(double scale)
{
    m_scale = scale;
}

void IInterpreter::SetOutputLayerName(const std::vector<cv::String>& vStrLayerName)
{
    m_vOutputLayerName.reserve(vStrLayerName.size());
    std::copy(vStrLayerName.begin(), vStrLayerName.end(), m_vOutputLayerName.begin());
}