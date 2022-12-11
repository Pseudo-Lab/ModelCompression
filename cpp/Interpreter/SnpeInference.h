#ifndef SNPEINFERENCE_H
#define SNPEINFERENCE_H

#include <iostream>
#include <string>

#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DiagLog/IDiagLog.hpp"

#include "DlSystem/PlatformConfig.hpp"

#include "opencv2/opencv.hpp"

#include "IInterpreter.h"

enum class BUFF_DATATYPE
{
    UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16
};
enum class BUFF_TYPE
{
    CPUBUFFER, GLBUFFER
};


class CSnpeInference : public IInterpreter
{
public:
    CSnpeInference(const std::string& strWeightFilePath, const std::string& strConfigFilePath);
    ~CSnpeInference();

    bool SetInputShape(int inWidth, int inHeight, int inChannels);
    bool SetDelegate(DELEGATE _delegate);
    bool LoadModel();

    std::unordered_map<std::string, cv::Mat> Interpret(const cv::Mat& srcImg);

private:

    bool Init(const std::string& strRuntimeType, const std::string& strBufferType);

    std::unique_ptr<zdl::DlSystem::ITensor> LoadInputTensor(const cv::Mat& srcImg);
    zdl::DlSystem::Runtime_t CheckRuntime(zdl::DlSystem::Runtime_t runtime, bool &staticQuantization);

    bool m_isRuntimeSpecified;
    bool m_isStaticQuantization;

    BUFF_DATATYPE m_bufferType;
    int m_bitWidth;

    BUFF_TYPE m_userBufferSourceType;

    std::unique_ptr<zdl::DlContainer::IDlContainer> m_pContainer;

    std::unique_ptr<zdl::SNPE::SNPE> m_pSnpe;
    zdl::DlSystem::PlatformConfig m_platformConfig;

    bool m_isUseUserSuppliedBuffers;
    int m_batchSize;

    zdl::DlSystem::Runtime_t m_runtime;
    zdl::DlSystem::RuntimeList m_runtimeList;

};

#endif //SNPEINFERENCE_H

