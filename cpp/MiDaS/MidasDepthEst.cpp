#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "MidasDepthEst.h"

using Time = std::chrono::steady_clock;
// *******************************************************************************



CMidasDepthEst::CMidasDepthEst(std::unique_ptr<IInterpreter> pInterpreter) :
                                m_isInvertScale(false)
{
    m_pInterpreter = std::move(pInterpreter);
    m_pInterpreter->LoadModel();
}

void CMidasDepthEst::SetInvertedScale(bool isInvertScale)
{
    m_isInvertScale = isInvertScale;
}

cv::Mat CMidasDepthEst::Run(const cv::Mat& srcImg)
{
    cv::Mat inputImg = (srcImg - MIDAS_DEFAULT_MEAN) / MIDAS_DEFAULT_STD;

    std::unordered_map<std::string, cv::Mat> vNetOutput = m_pInterpreter->Interpret(inputImg);

    // TODO: check output shape and method
    cv::Mat& outputTensor = vNetOutput["Identity"];
    cv::Mat depthMat;
    cv::resize(outputTensor, depthMat, srcImg.size(), 0, 0, cv::INTER_CUBIC);

    if (m_isInvertScale)
    {
        double min;
        double max;
        cv::minMaxIdx(depthMat, &min, &max);

        depthMat = (1 / depthMat) * max;
    }

    return std::move(depthMat);
}

cv::Mat CMidasDepthEst::DrawDepth(const cv::Mat& depthMat)
{
    cv::Mat normDepthMat = depthMat.clone();

    double min;
    double max;
    cv::minMaxIdx(depthMat, &min, &max);
    normDepthMat = (255 * (normDepthMat - min) / (max - min));

    normDepthMat.convertTo(normDepthMat, CV_8UC1);
    cv::applyColorMap(normDepthMat, normDepthMat, cv::COLORMAP_MAGMA);

    return std::move(normDepthMat);
}

void CMidasDepthEst::EvaluateNyuDepthV2(const std::string& strDbPath, const std::string& strSavePath, bool isInvertScale)
{
    double averageAbsRel = 0.0;
    double averageRmse = 0.0;
    double averageSiRmse = 0.0;

    int vaildCont = 0;
    for (int iter = 0; iter < NUM_NYUDEPTHV2_TEST_SET; iter++)
    {
        // Step 1. Load GT depth
        std::ostringstream ostr;
        ostr << strDbPath << "/gt/gt" << std::setfill('0') << std::setw(5)
             << std::to_string(iter) << ".csv";

        std::string path1 = ostr.str();

        std::cout << path1 << "\n";

        //*
        cv::Mat depthMat = LoadNyuDepthGtFromCsv(path1);

        ostr.str("");
        ostr.clear();

        // Step 2. Load source image
        ostr << strDbPath << "/img/source" << std::setfill('0') << std::setw(5)
             << std::to_string(iter) << ".jpg";

        std::string path2 = ostr.str();
        std::cout << "Input Image: " << path2 << "\n";

        cv::Mat srcImg = cv::imread(path2);

        // Step 3. Prediction
        cv::Mat predictMat = Run(srcImg);
        ostr.str("");
        ostr.clear();


        // Step 4. Save Result
        ostr << strSavePath << "/res_pred_gt" << std::setfill('0') << std::setw(5)
             << std::to_string(iter) << ".jpg";
        cv::Mat resMat;
        cv::hconcat(srcImg, DrawDepth(predictMat), resMat);
        cv::hconcat(resMat, DrawDepth(depthMat), resMat);
        cv::imwrite(ostr.str(), resMat);
        ostr.str("");
        ostr.clear();

        // Step 5. performance
        float absRel = GetAbsRel(predictMat, depthMat);
        float rmse = GetRmsError(predictMat, depthMat);
        float siRmse = GetSiRmsError(predictMat, depthMat);

        if(!isnan(siRmse))
        {
            vaildCont++;
            averageAbsRel += absRel;
            averageRmse += rmse;
            averageSiRmse += siRmse;

            std::cout << "DB " << iter << "-> " << "AbsRel: " << absRel << ", RMSE: " << rmse << ", Si-RMSE: " << siRmse
                      << "\n";
        }
    }

    averageAbsRel /= vaildCont;
    averageRmse /= vaildCont;
    averageSiRmse /= vaildCont;

    std::cout << "Average-> AbsRel: " << averageAbsRel << ", RMSE: " << averageRmse << ", Si-RMSE: " << averageSiRmse
              << "\n";
}
float CMidasDepthEst::GetAbsRel(const cv::Mat& predict, const cv::Mat& gt)
{
    cv::Scalar rel = cv::mean(cv::abs(gt - predict) / gt);
    return rel[0];
}

float CMidasDepthEst::GetRmsError(const cv::Mat& predict, const cv::Mat& gt)
{
    cv::Mat subMat = gt - predict;
    cv::Scalar mse = cv::mean(subMat.mul(subMat));

    return cv::sqrt(mse[0]);
}

template <typename T>
cv::Mat CMidasDepthEst::GetLogMat(const cv::Mat& srcImg)
{
	cv::Mat targetImg = srcImg.clone();
	
	T* dataPtr = (T*)targetImg.data;
	int widthStep = targetImg.step1(); 
	for(int k =0; k <srcImg.rows; k++)
	{
		int rowIdx = k* widthStep;
		for(int q =0; q <srcImg.cols; q++)
		{
			if(dataPtr[rowIdx +q] <1.0f)
				dataPtr[rowIdx +q] = 1.0f;
		}
	}

	cv::log(targetImg, targetImg);

	return std::move(targetImg);	
}

float CMidasDepthEst::GetSiRmsError(const cv::Mat& predict, const cv::Mat& gt)
{
    cv::Mat logPredict = GetLogMat<float>(predict);
    cv::Mat logGt = GetLogMat<float>(gt);

    cv::Mat logDiffMat = logPredict - logGt;
    cv::Scalar alpha = cv::sum(logDiffMat.mul(logDiffMat)) / (predict.cols * predict.rows);

    cv::Mat valMat = logGt - logPredict + alpha[0];
    cv::Scalar mse = cv::sum(valMat.mul(valMat)) / (predict.cols * predict.rows);

    return cv::sqrt(mse[0]);

}

cv::Mat CMidasDepthEst::LoadNyuDepthGtFromCsv(std::string& strGtPath)
{
    std::ifstream inputfile(strGtPath);
    std::string current_line;

    std::vector <std::vector<float>> all_data;

    while (getline(inputfile, current_line))
    {
        std::vector<float> values;
        std::stringstream temp(current_line);
        std::string single_value;
        while (getline(temp, single_value, ','))
        {
            values.push_back(atof(single_value.c_str()));
        }
        all_data.push_back(values);
    }

    cv::Mat depthMat = cv::Mat::zeros((int) all_data.size(), (int) all_data[0].size(), CV_32FC1);

    for (int rows = 0; rows < (int) all_data.size(); rows++)
    {
        for (int cols = 0; cols < (int) all_data[0].size(); cols++)
        {
            depthMat.at<float>(rows, cols) = all_data[rows][cols];
        }
    }

    return std::move(depthMat);
}


std::vector <std::string> CMidasDepthEst::GetLabel(const std::string& strPath)
{
    std::vector <std::string> vLabels;

    std::ifstream input(strPath);
    for (std::string line; std::getline(input, line);)
    {
        vLabels.push_back(line);
    }

    return std::move(vLabels);
}



/*
CTfLiteInterpreter::CTfLiteInterpreter(int inWidth, int inHeight, int inChannels)
{
    m_inWidth = inWidth;
    m_inHeight = inHeight;
    m_inChannels = inChannels;

    m_pModel = nullptr;
}

bool CTfLiteInterpreter::LoadModel(const std::string& strModelPath)
{
    // Step 1. Load Model
    m_pModel = tflite::FlatBufferModel::BuildFromFile(strModelPath.c_str());

    // Step 2. Build the interpreter
    tflite::InterpreterBuilder(*m_pModel.get(), m_resolver)(&m_pInterpreter);
    bool isModelOk = (m_pInterpreter->AllocateTensors() == kTfLiteOk);

    return isModelOk;
}

TfLiteTensor* CTfLiteInterpreter::Interpret(cv::Mat& image)
{

    // Step 3. Load Image
    memcpy(m_pInterpreter->typed_input_tensor<float>(0), image.data, image.total() * image.elemSize());

    // Step 5. Run inference
    m_pInterpreter->Invoke();

    image.release();

    // Step 6. Print Result
    return std::move(m_pInterpreter->tensor(m_pInterpreter->outputs()[0]));

}

cv::Mat CTfLiteInterpreter::ConvertInput(const cv::Mat& srcImg)
{
    cv::Mat image = srcImg.clone();

    if (m_inChannels == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    }
    else
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        image.convertTo(image, CV_32FC1, 1.0 / 255.0);
    }
    cv::resize(image, image, cv::Size(m_inWidth, m_inHeight));

    return std::move(image);
}

std::vector<int> CTfLiteInterpreter::GetSortedIndex(int numElement, float* pData)
{
    std::vector<int> vRank(numElement);
    std::iota(vRank.begin(), vRank.end(), 0);
    std::sort(vRank.begin(), vRank.end(), [&](int i, int j) { return pData[i] > pData[j]; });

    return std::move(vRank);
}//*/