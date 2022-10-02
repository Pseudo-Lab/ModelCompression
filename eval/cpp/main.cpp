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

/*
#include "face/CenterFace.h"
#include "face/MobileArcFace.h"
#include "face/Utils.h"

static const std::string CAM_DEVICE = "/dev/video0";

void FaceDetectionTest(std::unique_ptr<CCenterFace>& pFaceDetector);
int FaceVerification(std::unique_ptr<CCenterFace>& pFaceDetector, std::unique_ptr<CMobileArcFace>& pFaceRecognizer);

std::vector<cv::Vec3b> g_vColor = CUtils::GetColorSet(20);

enum class OP_MODE
{
    NONE = 0, VERIFICATION, CHANGE_MODE, REGISTRATION, EXIT
};

std::istream& operator>>(std::istream& is, OP_MODE& mode)
{
    int a;
    is >> a;

    if (a < 1 || a > 4)
    {
        a = 0;
    }

    mode = static_cast<OP_MODE>(a);

    return is;
}
//*/
int main(int argc, char** argv)
{

	std::unique_ptr<CDeepLabv3> pDeepLabv3 = std::make_unique<CDeepLabv3>("model.dlc");
	
	cv::Mat srcImg = cv::imread("/media/Data/lib/Git/ModelCompression/res/deeplab1.png");

    auto t1 = std::chrono::steady_clock::now();
    cv::Mat maskMat = pDeepLabv3->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

/*
    std::cout << "================================================\n";
    std::cout << "Detect " << vFaceLocation.size() << " face(s)\n";
    std::cout << "Detection Time: " << diff.count() << " [msec]\n";

    CUtils::DrawResult(srcImg, vFaceLocation, g_vColor);

    std::string strSavePath = "../res/output.png";
    cv::imwrite(strSavePath, srcImg);
    std::cout << "Save Path: " << strSavePath << "\n";
//*/	
	
/*
    std::unique_ptr<CCenterFace> pFaceDetector = std::make_unique<CCenterFace>();
    std::unique_ptr<CMobileArcFace> pFaceRecognizer = std::make_unique<CMobileArcFace>();

    if (argc < 2 || strcmp("-fd", argv[1]))
    {
        FaceDetectionTest(pFaceDetector);
    }
    else if (strcmp("-fv", argv[1]))
    {
        FaceVerification(pFaceDetector, pFaceRecognizer);
    }
//*/
	std::cout << "Hello World!!\n";
    return 0;
}
/*
int FaceVerification(std::unique_ptr<CCenterFace>& pFaceDetector, std::unique_ptr<CMobileArcFace>& pFaceRecognizer)
{
    std::string strCamPipeline = "qtiqmmfsrc ! video/x-h264,format=NV12,width=1920,height=1080,framerate=30/1 ! h264parse ! mp4mux ! queue ! appsink";
    cv::VideoCapture cap(strCamPipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "[Error] Can't open camera device!!." << std::endl;
        return -1;
    }

    while (1)
    {

        OP_MODE opMode;
        bool isRegistrationMode = pFaceRecognizer->CheckRegistrationMode();

        std::cout << "[MODE: " << ((isRegistrationMode) ? "Registration" : "Verification") << "]\n";
        std::cout << "Select Next Action: \n 1. verification \n 2. Change Mode `\n 3. Register Face \n 4. Exit \n>> ";
        std::cin >> opMode;

        // Wrong input
        if (opMode == OP_MODE::NONE)
        {
            continue;
        }
        // EXIT
        if (opMode == OP_MODE::EXIT)
        {
            goto EXIT;
        }
        // Change Mode
        if (opMode == OP_MODE::CHANGE_MODE)
        {
            (isRegistrationMode) ? pFaceRecognizer->SetVerificationMode() : pFaceRecognizer->SetRegistrationMode();
            continue; // TODO check operate in while loop
        }

        // Step 1. Capture
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty())
        {
            std::cerr << "Failure to Capture";
            return -1;
        }

        // Step 3.1. Face Localization / Tracking
        std::vector<DetectionBox<int>> vFaceLocation = pFaceDetector->Run(frame);

        if (opMode == OP_MODE::REGISTRATION && isRegistrationMode && vFaceLocation.size() != 1)
        {
            vFaceLocation[0].faceId = pFaceRecognizer->Register(frame, vFaceLocation[0].bbox);
            if (vFaceLocation[0].faceId != -99)
            {
                std::cout << "Registered!\n";
            }
            else
            {
                std::cout << "Already registered person!\n";
            }
        }
        else
        {
            // TODO: whole image input
            pFaceRecognizer->Run(frame, vFaceLocation);

            for (auto& result : vFaceLocation)
            {
                std::cout << "verified ID: " << result.faceId << "\n";
            }
        }
    }

    EXIT:
    std::cout << "Terminate!\n";
}


void FaceDetectionTest(std::unique_ptr<CCenterFace>& pFaceDetector)
{
    cv::Mat srcImg = cv::imread("../res/000388.jpg");

    auto t1 = std::chrono::steady_clock::now();
    std::vector<DetectionBox<int>> vFaceLocation = pFaceDetector->Run(srcImg);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << "================================================\n";
    std::cout << "Detect " << vFaceLocation.size() << " face(s)\n";
    std::cout << "Detection Time: " << diff.count() << " [msec]\n";

    CUtils::DrawResult(srcImg, vFaceLocation, g_vColor);

    std::string strSavePath = "../res/output.png";
    cv::imwrite(strSavePath, srcImg);
    std::cout << "Save Path: " << strSavePath << "\n";
}
//*/
