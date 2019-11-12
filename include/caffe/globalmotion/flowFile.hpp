#ifndef FLOWFILE_HPP
#define FLOWFILE_HPP

#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>

class FlowFile
{
public:
	FlowFile(){}	
	FlowFile(const std::string uFlowPath, const std::string vFlowPath);
	FlowFile(cv::Mat &flow);
	~FlowFile(){}

	
	void changeSize(int r);

	cv::Mat uFlow, vFlow;
	int height, width;
};

#endif
