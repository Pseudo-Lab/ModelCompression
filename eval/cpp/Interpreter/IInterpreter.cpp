#include "IInterpreter.h"

void IInterpreter::SetInputMean(cv::Scalar value)
{
    m_Mean = value;
}

void IInterpreter::SetInputScale(double scale)
{
    m_scale = scale;
}

void IInterpreter::SetInputOrderRgb(bool isOrderRgb)
{
    m_isOrderRgb = isOrderRgb;
}

void IInterpreter::SetOutputLayerName(const std::vector<cv::String>& vStrLayerName)
{
    m_vOutputLayerName.reserve(vStrLayerName.size());
    std::copy(vStrLayerName.begin(), vStrLayerName.end(), m_vOutputLayerName.begin());
}

std::vector<cv::String> IInterpreter::GetOutputLayerName()
{
    std::vector<cv::String> vOutputLayerName(m_vOutputLayerName.size());
    std::copy(m_vOutputLayerName.begin(), m_vOutputLayerName.end(), vOutputLayerName.begin());

    return std::move(vOutputLayerName);
}