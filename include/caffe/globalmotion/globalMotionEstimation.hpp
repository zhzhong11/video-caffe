#ifndef GLOBALMOTIONESTIMATION_HPP
#define GLOBALMOTIONESTIMATION_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#include <math.h>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "caffe/globalmotion/inputFilePath.hpp"
#include "caffe/globalmotion/flowFile.hpp"
#include "caffe/globalmotion/GMEParameter.hpp"

void calculateRatio(const std::string &preIMGPath, const std::string &curIMGPath, float &hRatio, float &wRatio);
void calculateRatio(cv::Mat preImg, cv::Mat curImg, float &hRatio, float &wRatio);
float defineMinority(cv::Mat &STGS);
void showMask(cv::Mat data);
void geneuratePosition(cv::Mat &pos, int h, int w, bool rowflag);
void extractOpticalFlow(cv::Mat &preImg,cv::Mat &curImg,cv::Mat &flowOut);
void flowTomotion(const cv::Mat &flow,cv::Mat &motion,const int blockHeight=8, const int blockWidth=8);


struct Point
{
    int x;
    int y;
};

int matchErr(const cv::Mat &preImg, const cv::Mat &curImg, int preX, int preY, int curX, int curY, const int blockHeight=8, const int blockWidth=8);
void TSSSearch(const cv::Mat &preImg, const cv::Mat &curImg, int preX, int preY, int curX, int curY,Point &moveVector, const int searchLength=4, const int blockHeight=8, const int blockWidth=8);
void motionEstimation(const cv::Mat &preImg, const cv::Mat &curImg, cv::Mat &motionVector,
    const int searchLength=4, const int blockHeight=8, const int blockWidth=8);

void diamondSearch(const cv::Mat &preImg, const cv::Mat &curImg, int preX, int preY, int curX, int curY,Point &moveVector,  const int blockHeight=8, const int blockWidth=8);
class GMEParameter;

/*class GlobalMotionEstimation:
	calculate motionvector from input path and get the GME parameters
	Initiated with the rootPath and a ratio for downsampling*/
class GlobalMotionEstimation
{
public:
	friend class GMEParameter;
	friend class GMEParameter2;
	friend class GMEParameter6;

	
	GlobalMotionEstimation(const std::string imgPath,const std::string flowPath="",
							 const std::string savePath="", const int r = 1);
	GlobalMotionEstimation(const std::vector<cv::Mat> &img,const int r = 1);
	~GlobalMotionEstimation();
	
	void calculateParameter();

	std::vector<cv::Mat> parameter;
	//std::vector<cv::Mat> mask;
	//cv::Mat parameter;
	cv::Mat mask;
	std::vector<cv::Mat> motionVector;
	std::vector<cv::Mat> subMotion;

private:
	int initial();
	void calculateMotionVector(const int i);
	void subMotionVector();
	void checkRatio(const int i);
	void saveMat(const int i);
	
	InputFilePath inputIMG;
	InputFilePath inputFlow;
	FlowFile flow;
	GMEParameter *GMEModel;
	
	unsigned int nbframes;
	int height, width, ratio;
	std::string imgPath_;
	std::string flowPath_;
	std::string savePath_;
	// Choice the input way, false-dir input, true-vector<Mat>
	bool flag;
	std::vector<cv::Mat> img_;
	cv::Mat xPos, yPos;
};

#endif
