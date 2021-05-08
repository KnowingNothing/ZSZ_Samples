//============================================================================
// Name        : main.cpp
// Author      : copper
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "cnn.h"
#include <iostream>
using namespace std;

// #define TEST_PREDICT
#ifdef  TEST_PREDICT
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
int test_CNN_predict()
{
	CNN cnn2;
	cnn2.init();
	bool flag = cnn2.readModelFile("cnn.model");
	if (!flag) {
		std::cout << "read cnn model error" << std::endl;
		return -1;
	}

	int width{ 32 }, height{ 32 };
	std::vector<int> target{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	std::string image_path{ "images/" };

	for (auto i : target) {
		std::string str = std::to_string(i);
		str += ".png";
		str = image_path + str;

		cv::Mat src = cv::imread(str, 0);
		if (src.data == nullptr) {
			fprintf(stderr, "read image error: %s\n", str.c_str());
			return -1;
		}

		cv::Mat tmp(src.rows, src.cols, CV_8UC1, cv::Scalar::all(255));
		cv::subtract(tmp, src, tmp);

		cv::resize(tmp, tmp, cv::Size(width, height));

                std::cout << tmp.data << std::endl;
		auto ret = cnn2.predict(tmp.data, width, height);
		cnn2.saveMiddlePic(i);
		fprintf(stdout, "the actual digit is: %d, correct digit is: %d\n", ret, i);
	}

	return 0;
}
#endif

int main() {
#ifdef  TEST_PREDICT
	test_CNN_predict();
#else
	CNN Tcnn;
    Tcnn.init();
    Tcnn.train();
#endif
	cout << "!!!Hello ^_^ ** ^_^ ** ^_^!!!" << endl; // prints !!!Hello World!!!
	return 0;
}

