#ifndef GMEPARAMETER_HPP
#define GMEPARAMETER_HPP

#include "caffe/globalmotion/globalMotionEstimation.hpp"

/*class gmeParameters:
	calculate GME parameters and as an sub-process of GlobalMotionEstimation
	Initiated with a reference of GlobalMotionEstimation*/

class GlobalMotionEstimation;
class GMEParameter
{
protected:
	friend class GlobalMotionEstimation;
	GMEParameter(){}
	GMEParameter(GlobalMotionEstimation* gme);
	virtual ~GMEParameter(){}
	
	virtual void calculateParameter() = 0;
	virtual bool checkIfStop() = 0;

	void getParameter();
	int initial();	
	void updateMask();
	void errorHist(const cv::Mat &diffPos);

	GlobalMotionEstimation *GME;

	cv::Mat xCurPos, yCurPos, xCalPos, yCalPos, preParameter, parameter, mask;
	std::vector<float> diffVar;

	int reGetCnt, height, width;
	float parDiff1, sigma1, parDiff2, sigma2;
	bool breakFlag;
};

class GMEParameter2 : public GMEParameter
{
public:
	GMEParameter2(GlobalMotionEstimation *gme) : GMEParameter(gme) { }
	bool checkIfStop();
	void calculateParameter();
};

class GMEParameter6 : public GMEParameter
{
public:
	GMEParameter6(GlobalMotionEstimation *gme) : GMEParameter(gme) { }
	bool checkIfStop();
	void calculateParameter();
};

#endif
