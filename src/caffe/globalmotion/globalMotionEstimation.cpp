#include "caffe/globalmotion/globalMotionEstimation.hpp"
#include <opencv2/legacy/legacy.hpp> 
#include <future>

/*s: contains flow Directory and frame images
	<rootPath> --- frame images
			=>flow Directory <rootPath>/flow/
			 --- flow files
ratio: default ratio = 1*/
GlobalMotionEstimation::GlobalMotionEstimation(const std::string imgPath,const std::string flowPath, const std::string savePath, const int r) 
											   : ratio(r), imgPath_(imgPath), flowPath_(flowPath), savePath_(savePath), flag(false)
{
	
	checkInPutDir(imgPath_);

	checkInPutDir(savePath_);
	inputIMG = InputFilePath(imgPath_, "jpg");
	if(flowPath_!=""){
		checkInPutDir(flowPath_);
		inputFlow = InputFilePath(flowPath_, "jpg");
	}
	//parameter = std::vector<cv::Mat>();
	//mask = std::vector<cv::Mat>();
}
GlobalMotionEstimation::GlobalMotionEstimation(const std::vector<cv::Mat> &img,const int r)
												:ratio(r), flag(true), img_(img)
{
	parameter = std::vector<cv::Mat>();
	//mask = std::vector<cv::Mat>();
}

/*Check the files in rootPath and get file lists of image and flow*/
int GlobalMotionEstimation::initial()
{
	// initial file info
	// members:
	// std::map<std::string, std::string> filePath
	// std::vector<std::string> fileName
	if(!flag){
		if (inputIMG.initial())
			return 1;
		if (flowPath_!=""&&inputFlow.initial())
			return 1;
		if (flowPath_!=""&&inputIMG.numFiles != inputFlow.numFiles/2 + 1)
		{
			std::cerr << "Check the NUM of the files, match failed: IMAGE - " << inputIMG.numFiles << " FLOW - " << inputFlow.numFiles << std::endl;
			return 1;
		}else{
			nbframes = inputIMG.numFiles;
			return 0;	
		}
	}else
	{
		if(img_.empty())
		{
			std::cerr << "Check the input Mat, it is empty."<<std::endl;
			return 1;
		}else
		{	
			nbframes = img_.size();
			return 0;
		}
		

	}
	
	
}

/*	get GME parameters of motionvector of each frame with GMEParameter*/
void GlobalMotionEstimation::calculateParameter()
{
	if (initial())
		exit(1);
	
	for (int i = 1; i <nbframes; ++i)
	{
		calculateMotionVector(i);
		GMEModel->getParameter();
		//mask.push_back(GMEModel->mask);
		mask=(GMEModel->mask).clone();
		parameter.push_back(GMEModel->parameter);
		//parameter=(GMEModel->parameter).clone();
		//showMask(GMEModel->mask);
		//saveMat(i);
		delete GMEModel;
	}
	subMotionVector();
}

/*	read the flowfile and refresh the motionvector, refresh gp with a pointer of gmeParameters for calculating of GME parameters
cnt: input,the index of frames for processing */
void GlobalMotionEstimation::calculateMotionVector(const int cnt)
{	
	if(!flag)
	{
		if(flowPath_!=""){
			char flow_filename_horz[256];
			char flow_filename_vert[256];
			snprintf(flow_filename_horz, sizeof(flow_filename_horz), "%simage_%04d.jpg",
               		flowPath_.c_str(), 2*cnt-1);
    		snprintf(flow_filename_vert, sizeof(flow_filename_vert), "%simage_%04d.jpg",
               		flowPath_.c_str(), 2*cnt);
			//std::string flowPath(inputFlow.filePath.at(inputFlow.fileName[cnt]));
			flow = FlowFile(flow_filename_horz,flow_filename_vert);
			//flow->initial();
		}else{
			char preImgName[256];
			char curImgName[256];
			snprintf(preImgName, sizeof(preImgName), "%simage_%04d.jpg",
               		imgPath_.c_str(), cnt);
    		snprintf(curImgName, sizeof(curImgName), "%simage_%04d.jpg",
               	imgPath_.c_str(), cnt+1);
			cv::Mat preIMG, curIMG;
			preIMG = cv::imread(preImgName, CV_LOAD_IMAGE_GRAYSCALE);
			curIMG = cv::imread(curImgName, CV_LOAD_IMAGE_GRAYSCALE);
	
			cv::Mat curFlow;
			extractOpticalFlow(preIMG,curIMG,curFlow);
			flow = FlowFile(curFlow);
		}
	}else
	{
		cv::Mat curFlow;
		extractOpticalFlow(img_[cnt-1],img_[cnt],curFlow);
		flow = FlowFile(curFlow);
		motionVector.push_back(curFlow);
	}
	
	if (cnt == 1)
	{
		//mask.push_back(cv::Mat::ones(flow.height, flow.width, CV_32F));
		mask=cv::Mat::ones(flow.height, flow.width, CV_32F);
		geneuratePosition(xPos, flow.height, flow.width, true);
		geneuratePosition(yPos, flow.height, flow.width, false);
	}
	checkRatio(cnt);
}

/*	check which method should be used and return with a pointer to gmeParameters2 or gmeParameters6
cnt: input,the index of frames for processing*/
void GlobalMotionEstimation::checkRatio(const int cnt)
{
	if (cnt == 1){
		GMEModel = new GMEParameter6(this);
		return;
	}
		

	float hRatio, vRatio;
	if(!flag){
		char preImgName[256];
		char curImgName[256];
		snprintf(preImgName, sizeof(preImgName), "%simage_%04d.jpg",
               	imgPath_.c_str(), cnt);
    	snprintf(curImgName, sizeof(curImgName), "%simage_%04d.jpg",
               	imgPath_.c_str(), cnt+1);
		calculateRatio(preImgName, curImgName, hRatio, vRatio);
	}else
	{
		calculateRatio(img_[cnt-1], img_[cnt], hRatio, vRatio);
	}
	
	

	if (hRatio >= 0.5 || vRatio >= 0.5){
		GMEModel = new GMEParameter2(this);
		return;
	}else{
		GMEModel = new GMEParameter6(this);
		return;
	}
		
}
void GlobalMotionEstimation::saveMat(const int cnt){

	cv::Mat test;
	//double testminv, testmaxv;
	// cv::normalize(data, test, 1, 0, cv::NORM_MINMAX);
	(GMEModel->mask).copyTo(test);
	test.convertTo(test, CV_8U, 255, 0);
	char saveImgName[256];
	snprintf(saveImgName, sizeof(saveImgName), "%simage_%04d.jpg",
               	savePath_.c_str(), cnt);
	std::cout<<"Save current image: "<<saveImgName<<std::endl;
	cv::imwrite(saveImgName,test);	
}
void GlobalMotionEstimation::subMotionVector(){
	int lenght=motionVector.size();
	subMotion.reserve(lenght);
	cv::Mat oneSubMotion;
	for(int i=0;i<lenght;++i){
		oneSubMotion=motionVector[i].clone();
		for(int h=0;h<oneSubMotion.rows;++h){
			float* motionData=oneSubMotion.ptr<float>(h);
			for(int w=0;w<oneSubMotion.cols;++w){
				motionData[2*w]+=parameter[i].at<float>(0)*w+parameter[i].at<float>(1)*h+parameter[i].at<float>(2)-w;
				motionData[2*w+1]+=parameter[i].at<float>(3)*w+parameter[i].at<float>(4)*h+parameter[i].at<float>(5)-h;
				
			}
		}
		subMotion.push_back(oneSubMotion);
		
	}
}
GlobalMotionEstimation::~GlobalMotionEstimation(){
		//delete GMEModel;
}
/*choose the method of GME calculation with 2-parameters or 6-parameters via calculating ratio
preImagePath: input,path of pre-frame image
curImagePath: input,path of current frame image
hRatio: output
vRatio: output*/
void calculateRatio(const std::string &preIMGPath, const std::string &curIMGPath, float &hRatio, float &vRatio)
{
	cv::Mat preIMG, curIMG, diffIMG, xGRDIMG, yGRDIMG;
	preIMG = cv::imread(preIMGPath, CV_LOAD_IMAGE_GRAYSCALE);
	curIMG = cv::imread(curIMGPath, CV_LOAD_IMAGE_GRAYSCALE);
	

	cv::GaussianBlur(preIMG, preIMG, cv::Size(5,5), 3, 3);
	cv::GaussianBlur(curIMG, curIMG, cv::Size(5,5), 3, 3);

	cv::subtract(curIMG, preIMG, diffIMG);
	diffIMG.convertTo(diffIMG, CV_32F);
	cv::Sobel(preIMG, xGRDIMG, CV_32F, 1,0,1,1,0, cv::BORDER_DEFAULT);
	cv::Sobel(preIMG, yGRDIMG, CV_32F, 0,1,1,1,0, cv::BORDER_DEFAULT);

	cv::Mat hSTGS = cv::Mat(preIMG.size(), CV_32F);
	cv::Mat vSTGS = cv::Mat(preIMG.size(), CV_32F);
	cv::divide(diffIMG, xGRDIMG, hSTGS);
	cv::divide(diffIMG, yGRDIMG, vSTGS);

	hRatio = defineMinority(hSTGS);
	vRatio = defineMinority(vSTGS);
}
void calculateRatio(cv::Mat preIMG, cv::Mat curIMG, float &hRatio, float &vRatio)
{
	cv::Mat  diffIMG, xGRDIMG, yGRDIMG;
	if(preIMG.channels()==3){
		cv::cvtColor(preIMG,preIMG,cv::COLOR_BGR2GRAY);
		cv::cvtColor(curIMG,curIMG,cv::COLOR_BGR2GRAY);
	}

	cv::GaussianBlur(preIMG, preIMG, cv::Size(5,5), 3, 3);
	cv::GaussianBlur(curIMG, curIMG, cv::Size(5,5), 3, 3);

	cv::subtract(curIMG, preIMG, diffIMG);
	diffIMG.convertTo(diffIMG, CV_32F);
	cv::Sobel(preIMG, xGRDIMG, CV_32F, 1,0,1,1,0, cv::BORDER_DEFAULT);
	cv::Sobel(preIMG, yGRDIMG, CV_32F, 0,1,1,1,0, cv::BORDER_DEFAULT);

	cv::Mat hSTGS = cv::Mat(preIMG.size(), CV_32F);
	cv::Mat vSTGS = cv::Mat(preIMG.size(), CV_32F);
	cv::divide(diffIMG, xGRDIMG, hSTGS);
	cv::divide(diffIMG, yGRDIMG, vSTGS);

	hRatio = defineMinority(hSTGS);
	vRatio = defineMinority(vSTGS);
}
/*defineMinority: calculate ratio of image difference and image gradient
STGS: input,the result of calculateRatio*/
float defineMinority(cv::Mat &STGS)
{
	int h = STGS.rows, w = STGS.cols;

	cv::Mat negativeMap, positiveMap, mulMap, powxMap, powyMap, xPos, yPos;
	cv::compare(STGS, 0, negativeMap, cv::CMP_LT);
	cv::compare(STGS, 0, positiveMap, cv::CMP_GT);

	unsigned int negNum = cv::sum(negativeMap).val[0];
	unsigned int posNum = cv::sum(positiveMap).val[0];

	cv::Mat &minMap(negNum > posNum ? positiveMap : negativeMap);

	geneuratePosition(xPos, h, w, true);
	geneuratePosition(yPos, h, w, false);

	minMap.convertTo(minMap,CV_32F,1.0/255.0,0);
	unsigned int minNum = cv::sum(minMap).val[0];
	cv::multiply(xPos, minMap, mulMap);
	int xCenter = cv::sum(mulMap).val[0] / minNum;
	cv::multiply(yPos, minMap, mulMap);
	int yCenter = cv::sum(mulMap).val[0] / minNum;

	cv::pow((xPos - xCenter), 2, powxMap);
	cv::pow((yPos - yCenter), 2, powyMap);

	cv::multiply(powxMap, minMap, mulMap);
	float varience = cv::sum(mulMap).val[0];
	cv::multiply(powyMap, minMap, mulMap);
	varience += cv::sum(mulMap).val[0];
	return varience / (minNum * minNum);
}

void showMask(cv::Mat data)
{
	// show the data in each interation
	cv::Mat test;
	double testminv, testmaxv;
	cv::namedWindow("GlobalMotionEstimation");
	// cv::normalize(data, test, 1, 0, cv::NORM_MINMAX);
	data.copyTo(test);
	test.convertTo(test, CV_8U, 255, 0);
	cv::minMaxIdx(data, &testminv, &testmaxv);
	std::cout << testminv << " " << testmaxv << std::endl;
	std::cout << cv::sum(test).val[0] << std::endl;
	cv::imshow("GlobalMotionEstimation", test);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

/*generate a defined metrix with positions values of each one
pos: the generated position matrix
h: input,height of metrix
w: input,width of metrix
rowflag: generate x-axis positions or y-axis positions
		 true for x-axis and false for y-axis*/
void geneuratePosition(cv::Mat &pos, int h, int w, bool rowflag)
{
	pos = cv::Mat::ones(h, w, CV_16U);
	if (rowflag)
	{
		for (int i = 0; i != w; i++)
			pos.col(i) *= i;
		pos -= w/2;
	}
	else{
		for (int i = 0; i != h; i++)
			pos.row(i) *= i;
		pos = h/2 - pos;
	}
	pos.convertTo(pos, CV_32F);
}

void extractOpticalFlow(cv::Mat &preImg,cv::Mat &curImg,cv::Mat &flowOut)
{
	cv::Mat flow,preGray,curGray;
	if(preImg.channels()==3){
		cv::cvtColor(preImg,preGray,cv::COLOR_BGR2GRAY);
		cv::cvtColor(curImg,curGray,cv::COLOR_BGR2GRAY);
   		cv::resize(preGray,preGray,cv::Size(224,224));
		cv::resize(curGray,curGray,cv::Size(224,224));
	}else
	{
		cv::resize(preImg,preGray,cv::Size(224,224));
		cv::resize(curImg,curGray,cv::Size(224,224));
		//preGray=preImg;
		//curGray=curImg;
	}
	//cv::calcOpticalFlowFarneback(preGray,curGray,flowOut,0.5,3,15,3,5,1.2,0);
	motionEstimation(preGray,curGray,flowOut);
	//cv::Mat motion;
	//flowTomotion(flow,motion);
	
	/*std::vector<cv::Mat> chann(2);
	cv::split(flow,chann);
	cv::normalize(chann[0],chann[0],0,255,cv::NORM_MINMAX);
	cv::normalize(chann[1],chann[1],0,255,cv::NORM_MINMAX);
	chann[0].convertTo(chann[0],CV_8UC1);
	chann[1].convertTo(chann[1],CV_8UC1);
	cv::merge(chann,flowOut);*/
	
}
int matchErr(const cv::Mat &preImg, const cv::Mat &curImg, int preX, int preY, int curX, int curY, 
    const int blockHeight, const int blockWidth)
{
    int blockErr=0;
    for(int i=0;i<blockHeight;++i){
        const uchar* preData=preImg.ptr<uchar>(preY+i);
        const uchar* curData=curImg.ptr<uchar>(curY+i);
        for(int j=0;j<blockWidth;++j){
            blockErr+=abs(curData[curX+j]-preData[preX+j]);
        }
    }
    return blockErr;
}
void TSSSearch(const cv::Mat &preImg, const cv::Mat &curImg, int preX, int preY, int curX, int curY,Point &moveVector,
    const int searchLength, const int blockHeight, const int blockWidth)
{
    std::vector<Point> searchPoint{{-searchLength,searchLength},{0,searchLength},{searchLength,searchLength},{-searchLength,0},
                                   {searchLength,0},{-searchLength,-searchLength},{0,-searchLength},{searchLength,-searchLength},{0,0}};
    
    int blockmatcherr=0;
    int flagMin=8;//记录当前匹配坐标
    int minMatch=matchErr(preImg,curImg,preX,preY,curX,curY,blockHeight,blockWidth);
    for(int i=0;i<8;++i){
        int searchPreX=preX+searchPoint[i].x;
        int searchPreY=preY+searchPoint[i].y;
        if(searchPreX>=0&&searchPreX<=preImg.cols-blockWidth&&searchPreY>=0&&searchPreY<=preImg.rows-blockHeight){
            blockmatcherr=matchErr(preImg,curImg,searchPreX,searchPreY,curX,curY,blockHeight,blockWidth);
        }else{
            blockmatcherr=-1;
        }
        if(blockmatcherr!=-1&&blockmatcherr<minMatch){
            minMatch=blockmatcherr;
            flagMin=i;
        }
    }
    moveVector.x-=searchPoint[flagMin].x;
    moveVector.y-=searchPoint[flagMin].y;
    if(searchLength==1){
        return;
    }else
    {
        TSSSearch(preImg,curImg,preX+searchPoint[flagMin].x,preY+searchPoint[flagMin].y,curX,curY, 
                    moveVector,searchLength/2,blockHeight,blockWidth);
    }
    
}

void motionEstimation(const cv::Mat &preImg, const cv::Mat &curImg, cv::Mat &motionVector,
    const int searchLength, const int blockHeight, const int blockWidth){
    if(preImg.type()!=CV_8UC1||curImg.type()!=CV_8UC1){
        std::cerr<<"Input Image must be Gray!"<<std::endl;
        return;
    }
    if(preImg.size()!=curImg.size()){
        std::cerr<<"preImg and curImg must have same size."<<std::endl;
        return;
    }

    int width=curImg.cols;
    int height=curImg.rows;
    motionVector=cv::Mat::zeros(height,width,CV_32FC2);
    int blockX=width/blockWidth;
    int blockY=height/blockHeight;
	//std::cout<<"blockX: "<<blockX<<std::endl;
	//std::cout<<"blockY: "<<blockY<<std::endl;
	//int blockY4=blockY/4;
	//std::future<void> ft1=async(std::launch::async,[&]{
    	for(int y=0;y<blockY;++y){
        	for(int x=0;x<blockX;++x){
            Point curVector{0,0};
            //TSSSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,searchLength,blockHeight,blockWidth);
			diamondSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,blockHeight,blockWidth);
           // std::cout<<curVector.x<<" "<<curVector.y<<std::endl;
            for(int i=y*blockHeight;i<y*blockHeight+blockHeight;++i){
				float* motionData=motionVector.ptr<float>(i);
                for(int j=x*blockWidth;j<x*blockWidth+blockWidth;++j){
                    //motionVector.at<float>(i,2*j)=curVector.x;
                    //motionVector.at<float>(i,2*j+1)=curVector.y;
					motionData[2*j]=curVector.x;
					motionData[2*j+1]=curVector.y;    
                }
            }
        }
		}
   /* }});
	std::future<void> ft2=async(std::launch::async,[&]{
    	for(int y=blockY4;y<2*blockY4;++y){
        	for(int x=0;x<blockX;++x){
            Point curVector{0,0};
            //TSSSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,searchLength,blockHeight,blockWidth);
			diamondSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,blockHeight,blockWidth);
           // std::cout<<curVector.x<<" "<<curVector.y<<std::endl;
            for(int i=y*blockHeight;i<y*blockHeight+blockHeight;++i){
				float* motionData=motionVector.ptr<float>(i);
                for(int j=x*blockWidth;j<x*blockWidth+blockWidth;++j){
                    //motionVector.at<float>(i,2*j)=curVector.x;
                    //motionVector.at<float>(i,2*j+1)=curVector.y;
					motionData[2*j]=curVector.x;
					motionData[2*j+1]=curVector.y;    
                }
            }
        }
    }});
	std::future<void> ft3=async(std::launch::async,[&]{
    	for(int y=2*blockY4;y<3*blockY4;++y){
        	for(int x=0;x<blockX;++x){
            Point curVector{0,0};
            //TSSSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,searchLength,blockHeight,blockWidth);
			diamondSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,blockHeight,blockWidth);
           // std::cout<<curVector.x<<" "<<curVector.y<<std::endl;
            for(int i=y*blockHeight;i<y*blockHeight+blockHeight;++i){
				float* motionData=motionVector.ptr<float>(i);
                for(int j=x*blockWidth;j<x*blockWidth+blockWidth;++j){
                    //motionVector.at<float>(i,2*j)=curVector.x;
                    //motionVector.at<float>(i,2*j+1)=curVector.y;
					motionData[2*j]=curVector.x;
					motionData[2*j+1]=curVector.y;    
                }
            }
        }
    }});
	std::future<void> ft4=async(std::launch::async,[&]{
    	for(int y=3*blockY4;y<blockY;++y){
        	for(int x=0;x<blockX;++x){
            Point curVector{0,0};
            //TSSSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,searchLength,blockHeight,blockWidth);
			diamondSearch(preImg,curImg,x*blockWidth,y*blockHeight,x*blockWidth,y*blockHeight,curVector,blockHeight,blockWidth);
           // std::cout<<curVector.x<<" "<<curVector.y<<std::endl;
            for(int i=y*blockHeight;i<y*blockHeight+blockHeight;++i){
				float* motionData=motionVector.ptr<float>(i);
                for(int j=x*blockWidth;j<x*blockWidth+blockWidth;++j){
                    //motionVector.at<float>(i,2*j)=curVector.x;
                    //motionVector.at<float>(i,2*j+1)=curVector.y;
					motionData[2*j]=curVector.x;
					motionData[2*j+1]=curVector.y;    
                }
            }
        }
    }});
	ft1.wait();
	ft2.wait();
	ft3.wait();
	ft4.wait();*/
}

void diamondSearch(const cv::Mat &preImg, const cv::Mat &curImg, int preX, int preY, int curX, int curY,Point &moveVector, const int blockHeight, const int blockWidth){
	int blockmatcherr=0;
	while(true){
		std::vector<Point> searchPoint{{-1,0},{1,0},{0,-1},{0,1},{0,0}};
		int minMatch=matchErr(preImg,curImg,preX,preY,curX,curY,blockHeight,blockWidth);
		int flagMin=4;//记录当前匹配坐标	
		for(int i=0;i<4;++i){
			int searchPreX=preX+searchPoint[i].x;
			int searchPreY=preY+searchPoint[i].y;
			if(searchPreX>=0&&searchPreX<=preImg.cols-blockWidth&&searchPreY>=0&&searchPreY<=preImg.rows-blockHeight){
				blockmatcherr=matchErr(preImg,curImg,searchPreX,searchPreY,curX,curY,blockHeight,blockWidth);
			}else{
				blockmatcherr=-1;
			}
			if(blockmatcherr!=-1&&blockmatcherr<minMatch){
           		minMatch=blockmatcherr;
            	flagMin=i;
        	}
		}
		if(flagMin!=4){
			preX+=searchPoint[flagMin].x;
			preY+=searchPoint[flagMin].y;
		}else{
			break;
		}
	}
	moveVector.x=curX-preX;
    moveVector.y=curY-preY;
}
void flowTomotion(const cv::Mat &flow,cv::Mat &motion,const int blockHeight,const int blockWidth){
	std::vector<cv::Mat> channels(2);
	cv::split(flow,channels);
	int width=channels[0].cols;
    int height=channels[0].rows;
    motion=cv::Mat::zeros(height,width,CV_32FC2);
    int blockX=width/blockWidth;
    int blockY=height/blockHeight;
	for(int y=0;y<blockY;++y){
		for(int x=0;x<blockX;++x){
			int sumBlockX=0;
			int sumBlockY=0;
			for(int i=y*blockHeight;i<y*blockHeight+blockHeight;++i){
				float* flowDatax=channels[0].ptr<float>(i);
				float* flowDatay=channels[1].ptr<float>(i);
                for(int j=x*blockWidth;j<x*blockWidth+blockWidth;++j){
					sumBlockX+=flowDatax[j];
					sumBlockY+=flowDatay[j];
				}
			}
			for(int i=y*blockHeight;i<y*blockHeight+blockHeight;++i){
				float* motionData=motion.ptr<float>(i);
                for(int j=x*blockWidth;j<x*blockWidth+blockWidth;++j){
					motionData[2*j]=sumBlockX/(blockHeight*blockWidth);
					motionData[2*j+1]=sumBlockY/(blockHeight*blockWidth);
				}
			}	
		}
	}
}
