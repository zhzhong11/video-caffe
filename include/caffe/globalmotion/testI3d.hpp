#ifndef TESTI3D_H_
#define TESTI3D_H_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "caffe/caffe.hpp"

using std::string;
using namespace caffe;
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class I3dClassifier {
 public:
  I3dClassifier(const string& model_file,
             const string& weights_file,
             const string& mean_file,
             const string& mean_value,
             const string& label_file="");

  int Classify(const std::vector<cv::Mat>& video, int N = 1);
  std::vector<float> Predict(const std::vector<cv::Mat>& video);

 private:
  void SetMean(const string& mean_file,const string& mean_value);

  void WrapInputLayer(int i,std::vector<cv::Mat >* input_channels);
  
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat >* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int temporary_length_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif
