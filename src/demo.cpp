// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "gms_matcher.h"
#include <chrono>
#include <iomanip>

void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(){
	Mat img1 = imread("../data/72.jpg");
	Mat img2 = imread("../data/326.jpg");

	// imresize(img1, 480);
	// imresize(img2, 480);

	GmsMatch(img1, img2);
}


int main()
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair();

	return 0;
}


void GmsMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	cv::ORB orb(1000);
	// orb->setFastThreshold(0);
	orb.detect(img1, kp1);
	orb.detect(img2, kp2);

	orb.compute(img1, kp1, d1);
	orb.compute(img2, kp2, d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif
	auto time_start = std::chrono::system_clock::now();

	// GMS filter
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, false);
	
	auto time_intr= std::chrono::system_clock::now();
	std::cout << "gms time: " << std::setprecision(6) << std::chrono::duration<double, std::milli>(time_intr - time_start).count() << " ms" << std::endl;
	
	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	imshow("show", show);
	waitKey();
}


