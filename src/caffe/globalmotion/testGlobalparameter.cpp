#include <pthread.h>
#include "caffe/caffe.hpp"
#include "caffe/globalmotion/testGlobalparameter.hpp"
#include "caffe/globalmotion/globalMotionEstimation.hpp"



using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

Classifier::Classifier(const string& model_file,
                       const string& weights_file,
                       const string& mean_file,
                       const string& mean_value,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1||num_channels_==2)
    << "Input layer should have 1 or 2 or 3 channels.";
  temporary_length_=input_layer->length();
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  Blob<float>* input_sixPara = net_->input_blobs()[1];
  CHECK_EQ(input_sixPara->channels(),temporary_length_-1);
  CHECK_EQ(input_sixPara->height(),6);


  /* Load the binaryproto mean file. */
  SetMean(mean_file,mean_value);

  /* Load labels. */
  if(!label_file.empty())
  {
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
      labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
      << "Number of labels is different from the output layer dimension.";
  }
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
int Argmax(const std::vector<float>& v) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + 1, pairs.end(), PairCompare);

  int result=pairs[0].second;
  
  return result;
}

/* Return the top N predictions. */
int Classifier::Classify(const std::vector<cv::Mat>& video, int N) {
  std::vector<float> output = Predict(video);
	
 /* for(int i=0;i<output.size();++i)
  {
	std::cout<<output[i]<<",";	
  }
	std::cout<<std::endl;*/
  //N = std::min<int>(labels_.size(), N);
  int max = Argmax(output);
  /*std::vector<Prediction> predictions;
  
  predictions.push_back(std::make_pair(labels_[max], output[max]));*/
  

  return max;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (std::getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    //cv::Mat pic_mean;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

std::vector<float> Classifier::Predict(const std::vector<cv::Mat>& video) {
  
  Blob<float>* input_layer = net_->input_blobs()[0];
  Blob<float>* input_sixPara = net_->input_blobs()[1];

  CHECK_EQ(video.size(),input_layer->length() )
    << "Number of video size is different from the input layer length.";

  input_layer->Reshape(1, num_channels_,temporary_length_,
                       input_geometry_.height, input_geometry_.width);
  std::vector<int> sixParaShape{1,temporary_length_-1,6};
  input_sixPara->Reshape(sixParaShape);
  /* Forward dimension change to all layers. */
  net_->Reshape();
  
  //std::vector<cv::Mat> input_channels;
  std::vector<std::vector<cv::Mat> > input_channels(input_layer->length());
  for(int i=0;i<input_layer->length();i++){
    WrapInputLayer(i,&input_channels[i]);
    Preprocess(video[i], &input_channels[i]);	
  }
//cal globalparameter time.
  float start_time=(float)cv::getTickCount();
  GlobalMotionEstimation result(video);
  result.calculateParameter();
  float end_time=(float)cv::getTickCount();
  std::cout<<"Cal globalparameter Runing times: "<<(end_time-start_time)/cv::getTickFrequency()<<std::endl;
  WrapInputLayer(result.parameter);
  const float *sixpara_data=input_sixPara->cpu_data();
  for(int i=0;i<12;++i)
  {
	std::cout<<sixpara_data[i]<<",";
  }
  std::cout<<std::endl;

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(int j,std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  int lenght= input_layer->length();
  float* input_data = input_layer->mutable_cpu_data()+width * height*j;
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height*lenght;
  }
}
void Classifier::WrapInputLayer(std::vector<cv::Mat >& input_sixparameter) {
  Blob<float>* input_layer = net_->input_blobs()[1];

  int timeLength=input_layer->channels();
  int sixPara=input_layer->height();
  float* input_data=input_layer->mutable_cpu_data(); 
  vector<cv::Mat > spilts;
  for (int i=0; i<timeLength;++i){
    cv::Mat channel(sixPara,1,CV_32FC1,input_data);
    spilts.push_back(channel);
    cv::split(input_sixparameter[i], spilts);
	//for(int j=0;j<sixPara;++j){
	//	*input_data++=input_sixparameter[i].at<float>(j);
	//}
    input_data+=sixPara;
	spilts.clear();
  }

}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  /*CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";*/
}



