#include "caffe/globalmotion/flowFile.hpp"

FlowFile::FlowFile(const std::string uFlowPath, const std::string vFlowPath)
{
	uFlow = cv::imread(uFlowPath, CV_LOAD_IMAGE_GRAYSCALE);
	vFlow = cv::imread(vFlowPath, CV_LOAD_IMAGE_GRAYSCALE);
	if(uFlow.empty() || vFlow.empty()){
		std::cerr << "Initial Failed, Please check the input file path: \"" 
		<< uFlowPath<<" or "<<vFlowPath<< "\"." << std::endl;
		return;
	}
	width = uFlow.cols;
	height = uFlow.rows;
	if (width <= 0 || height <= 0 || width >= 9999 || height >= 9999)
	{
		std::cerr << "Height: " << height << ", Width: " << width << ", Exclude the range(1-9999)." << std::endl;
		return ;
	}
	uFlow.convertTo(uFlow, CV_32F);
	vFlow.convertTo(vFlow, CV_32F);
}
FlowFile::FlowFile(cv::Mat &flow){
	if(flow.channels()!=2){
		std::cerr << "Check your flow. Flow must be two channels, but your flow's channels are "
		<<flow.channels()<<".\n";
	}
	std::vector<cv::Mat> chann(2);
	cv::split(flow,chann);
	uFlow=chann[0];
	vFlow=chann[1];
	width = uFlow.cols;
	height = uFlow.rows;
	
	uFlow.convertTo(uFlow, CV_32F);
	vFlow.convertTo(vFlow, CV_32F);
}

void FlowFile::changeSize(int ratio)
{
	if (ratio > 1)
	{
		ratio = ratio / 2 * 2;
		cv::pyrDown(uFlow, uFlow, cv::Size(width/ratio, height/ratio));
		cv::pyrDown(vFlow, vFlow, cv::Size(width/ratio, height/ratio));
		width = uFlow.cols;
		height = vFlow.rows;

		cv::medianBlur(uFlow, uFlow, 3);
		cv::medianBlur(vFlow, vFlow, 3);
	}else{
		std::cerr << "Please Re-Confirm the ratio, it should be larger than one." << std::endl;
		std::cout << "Continue without nothing changed or exit. Y(y) or N(n)." << std::endl;
		std::string ans;
		std::cin >> ans;
		if (ans != "Y" && ans != "y")
			exit(1);
	}
}
