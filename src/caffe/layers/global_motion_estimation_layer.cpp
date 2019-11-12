#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/global_motion_estimation_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/globalmotion/globalMotionEstimation.hpp"
namespace caffe {

template <typename Dtype>
GlobalMotionEstimationLayer<Dtype>::~GlobalMotionEstimationLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
GlobalMotionEstimationLayer<Dtype>::GlobalMotionEstimationLayer(const LayerParameter& param)
  :BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_(){
     for(int i=0;i<prefetch_.size();++i){
        prefetch_[i].reset(new globalBatch<Dtype>());
        prefetch_free_.push(prefetch_[i].get());
     }
  }
template <typename Dtype>
void GlobalMotionEstimationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom,top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
 for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
    prefetch_[i]->globalParameter_.mutable_cpu_data();
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
      prefetch_[i]->globalParameter_.mutable_cpu_data();
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";

}

template <typename Dtype>
void GlobalMotionEstimationLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
      bottom, const vector<Blob<Dtype>*>& top) {
  VideoDataParameter video_data_param=this->layer_param_.video_data_param();
  batch_size_ = video_data_param.batch_size();
  new_length_ = video_data_param.new_length();
  new_height_ = video_data_param.new_height();
  new_width_  = video_data_param.new_width();
  is_color_  = video_data_param.is_color();
  is_flow_   = video_data_param.is_flow();
  root_folder_ = video_data_param.root_folder();
  gap_example_=video_data_param.gap_example();
  
  const int num_ignore_example_label=video_data_param.ignore_example_label_size();
  if(num_ignore_example_label>0){   
      for(int i=0;i<num_ignore_example_label;++i){
          ignore_example_list_.push_back(video_data_param.ignore_example_label(i));
      }
  }

  CHECK((new_height_ == 0 && new_width_ == 0) ||
      (new_height_ > 0 && new_width_ > 0)) << "Current implementation requires "
      "new_height_ and new_width_ to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.video_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int frame_num, label;
  while (infile >> filename >> frame_num >> label) {
    triplet video_and_label;
    video_and_label.first = filename;
    video_and_label.second = frame_num;
    video_and_label.third = label;
    lines_.push_back(video_and_label);
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleVideos();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.video_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " video chunks.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a video clip, and use it to initialize the top blob.
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result;
  if(is_flow_){
    read_video_result = ReadFlowToCVMat(root_folder_ +
                                            lines_[lines_id_].first,
                                            lines_[lines_id_].second,
                                            new_length_, new_height_, new_width_,
                                            is_color_,
                                            &cv_imgs);
  }else{
    read_video_result = ReadVideoToCVMat(root_folder_ +
                                            lines_[lines_id_].first,
                                            lines_[lines_id_].second,
                                            new_length_, new_height_, new_width_,
                                            is_color_,
                                            &cv_imgs,1);
  }
  CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                              " at frame " << lines_[lines_id_].second << ".";
  CHECK_EQ(cv_imgs.size(), new_length_) << "Could not load " <<
                                          lines_[lines_id_].first <<
                                          " at frame " <<
                                          lines_[lines_id_].second <<
                                          " correctly.";
  // Use data_transformer to infer the expected blob shape from a cv_image.
  const bool is_video = true;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                  is_video);
  transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  
  CHECK_GT(batch_size_, 0) << "Positive batch size required";
  top_shape[0] = batch_size_;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
      << top[0]->shape(1) << "," << top[0]->shape(2) << ","
      << top[0]->shape(3) << "," << top[0]->shape(4);
  // label
  vector<int> label_shape(1, batch_size_);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }

  //globalParameter
  vector<int> globalParamter_shape(5);
  globalParamter_shape[0]=batch_size_;
  globalParamter_shape[1]=2;
  globalParamter_shape[2]=new_length_-1;
  globalParamter_shape[3]=224;
  globalParamter_shape[4]=224;
  top[2]->Reshape(globalParamter_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->globalParameter_.Reshape(globalParamter_shape);
  }
  
  
}

template <typename Dtype>
void GlobalMotionEstimationLayer<Dtype>::ShuffleVideos() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void GlobalMotionEstimationLayer<Dtype>::load_batch(globalBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  
 
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_globalParameter=batch->globalParameter_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    
    int gap_example=gap_example_;
    if(!ignore_example_list_.empty()){
	    for(int i=0;i<ignore_example_list_.size();++i){
	      if(lines_[lines_id_].third== ignore_example_list_[i]){
		    gap_example=1;
		    break;
	      }
	    }
    }
    
    std::vector<cv::Mat> cv_imgs;
    bool read_video_result ;
    if(is_flow_){
    read_video_result = ReadFlowToCVMat(root_folder_ +
                                            lines_[lines_id_].first,
                                            lines_[lines_id_].second,
                                            new_length_, new_height_, new_width_,
                                            is_color_,
                                            &cv_imgs);
  }else{
    read_video_result = ReadVideoToCVMat(root_folder_ +
                                            lines_[lines_id_].first,
                                            lines_[lines_id_].second,
                                            new_length_, new_height_, new_width_,
                                            is_color_,
                                            &cv_imgs,gap_example);
  }
    CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                                " at frame " << lines_[lines_id_].second << ".";
    CHECK_EQ(cv_imgs.size(), new_length_) << "Could not load " <<
                                             lines_[lines_id_].first <<
                                            " at frame " <<
                                            lines_[lines_id_].second <<
                                            " correctly.";
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    const bool is_video = true;
    this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
                                       is_video);

    //Calculate six global motiom estimation paramter.
    GlobalMotionEstimation sixGlobalParamter(cv_imgs);
    sixGlobalParamter.calculateParameter();
    //int offset_globalpar=batch->globalParameter_.offset(item_id);
	//prefetch_globalParameter+=offset_globalpar;
    /*for(int l=0;l<new_length_-1;++l){
	  Dtype* prefetch_globalParameter=batch->globalParameter_.mutable_cpu_data()+224*224*l;
      cv::Mat motionVector;
      extractOpticalFlow(cv_imgs[l],cv_imgs[l+1],motionVector);
	  vector<cv::Mat > spilts;
      for(int c=0;c<2;++c){
		cv::Mat channel(224,224,CV_32FC1,prefetch_globalParameter);
		spilts.push_back(channel);
      	prefetch_globalParameter +=224*224*(new_length_-1);
	  }
      cv::split(motionVector, spilts);
	  spilts.clear();
        //prefetch_globalParameter[offset_globalpar++]=static_cast<Dtype>(sixGlobalParamter.parameter[l].at<float>(p));
      
    }*/
	int offset_globalpar=batch->globalParameter_.offset(item_id); 
	Dtype* batchMotionData=prefetch_globalParameter+offset_globalpar;
	for(int l=0;l<new_length_-1;++l){
		Dtype* oneMotionData=batchMotionData+224*224*l;
      	//cv::Mat motionVector;
		//extractOpticalFlow(cv_imgs[l],cv_imgs[l+1],motionVector);
		for(int i=0;i<224;++i){
			//const float* motionData=motionVector.ptr<float>(i);
			const float* motionData=sixGlobalParamter.subMotion[l].ptr<float>(i);
			for(int j=0;j<224;++j){
				oneMotionData[i*224+j]=static_cast<Dtype>(motionData[2*j]);
				oneMotionData[i*224+j+224*224*(new_length_-1)]=static_cast<Dtype>(motionData[2*j+1]);
			}
		}
	}

    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].third;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleVideos();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void GlobalMotionEstimationLayer<Dtype>::InternalThreadEntry(){
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      globalBatch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        batch->globalParameter_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif    

}

template <typename Dtype>
void GlobalMotionEstimationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
  top[2]->ReshapeLike(prefetch_current_->globalParameter_);
  top[2]->set_cpu_data(prefetch_current_->globalParameter_.mutable_cpu_data());
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(GlobalMotionEstimationLayer, Forward);
#endif

INSTANTIATE_CLASS(GlobalMotionEstimationLayer);
REGISTER_LAYER_CLASS(GlobalMotionEstimation);

}  // namespace caffe
#endif  // USE_OPENCV