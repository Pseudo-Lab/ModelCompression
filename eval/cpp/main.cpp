#include <cstring>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <chrono>

#include "opencv2/opencv.hpp"


#include "DeepLabv3.h"

int main(int argc, char** argv)
{

    std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>();
	
    cv::Mat srcImg = cv::imread("../../../res/deeplab1.png");

    auto t1 = std::chrono::steady_clock::now();
    cv::Mat maskMat = pDeepLabv3->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    
    
    cv::imwrite("result.png", maskMat);
    
    return 0;
}
