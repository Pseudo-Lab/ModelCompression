#include <memory>

#include <chrono>

#include "opencv2/opencv.hpp"

#include "DeepLabv3.h"

int main(int argc, char** argv)
{
    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>();

    //* Test
    cv::Mat srcImg = cv::imread("../../../res/deeplab1.png");

    auto t1 = std::chrono::steady_clock::now();
    cv::Mat maskMat = pDeepLabv3->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    cv::imwrite("../res/result.png", maskMat);
    //*/

    // PASCAL 2012
#if BOARD
    std::string strVocPath = "datasets/VOCdevkit/VOC2012";
#else
    std::string strVocPath = "../../datasets/VOCdevkit/VOC2012";
#endif
    pDeepLabv3->EvaluateVOC12Val(strVocPath);

    return 0;
}
