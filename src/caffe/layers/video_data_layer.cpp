#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
      bottom, const vector<Blob<Dtype>*>& top) {
  VideoDataParameter video_data_param=this->layer_param_.video_data_param();
  batch_size_ = video_data_param.batch_size();
  new_length_ = video_data_param.new_length();
  new_height_ = video_data_param.new_height();
  new_width_  = video_data_param.new_width();
  is_color_  = video_data_param.is_color();
  is_flow_   = video_data_param.is_flow();
  is_avg_    = video_data_param.is_avg();
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
  }else if(is_avg_){
    read_video_result =ReadVideoToCVMat(root_folder_+lines_[lines_id_].first,
                                        new_height_,new_width_,is_color_,
                                        cv_imgs);
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
  if(!is_avg_)
  	CHECK_EQ(cv_imgs.size(), new_length_) << "Could not load " <<
                                          lines_[lines_id_].first <<
                                          " at frame " <<
                                          lines_[lines_id_].second <<
                                          " correctly.";
  // Use data_transformer to infer the expected blob shape from a cv_image.
  const bool is_video = true;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                  is_video);
  if(is_avg_)
	top_shape[2]=new_length_;
  this->transformed_data_.Reshape(top_shape);
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
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
 
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

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
  }else if(is_avg_){
    read_video_result =ReadVideoToCVMat(root_folder_+lines_[lines_id_].first,
                                        new_height_,new_width_,is_color_,
                                        cv_imgs);
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
	if(!is_avg_)
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
    //Same the whole video at equal intervals.
    if(is_avg_){
      std::vector<cv::Mat> equal_intervals_imgs(new_length_);
      int total_length=cv_imgs.size();
      int gap=total_length/new_length_;
      CHECK_GT(gap,0)<<"Your video less than the length, the file is "<<lines_[lines_id_].first;
      int redundant_imgs=(gap+1)*new_length_-total_length;
      int num_redundant=0;
      for(int l=0;l<new_length_;++l){
        int real_gap=gap;
        if(l>=redundant_imgs){
          real_gap=gap+1;
          //++num_redundant;
        }
        if(real_gap==1){
          equal_intervals_imgs[l]=cv_imgs[l*gap].clone();
        }else{
          equal_intervals_imgs[l]=cv_imgs[l*gap+num_redundant].clone();
          equal_intervals_imgs[l].convertTo(equal_intervals_imgs[l],CV_32F);
          for(int g=1;g<real_gap;++g){
            accumulate(cv_imgs[l*gap+g+num_redundant],equal_intervals_imgs[l]);
          }
          equal_intervals_imgs[l]/=real_gap;
          equal_intervals_imgs[l].convertTo(equal_intervals_imgs[l],CV_8U);
          if(l>=redundant_imgs){
            //real_gap=gap+1;
            ++num_redundant;
          }
        }
      }
      this->data_transformer_->Transform(equal_intervals_imgs, &(this->transformed_data_),
                                       is_video);
    }else{
      this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
                                       is_video);
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

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
#endif  // USE_OPENCV



