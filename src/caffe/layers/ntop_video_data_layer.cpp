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
#include "caffe/layers/ntop_video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
namespace caffe{

template <typename Dtype>
nTopVideoDataLayer<Dtype>::~nTopVideoDataLayer(){
    this->StopInternalThread();
}

template <typename Dtype>
nTopVideoDataLayer<Dtype>::nTopVideoDataLayer(const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_(){
     for(int i=0;i<prefetch_.size();++i){
        prefetch_[i].reset(new nTopBatch<Dtype>(this->layer_param_.video_data_param().top_size()));
        prefetch_free_.push(prefetch_[i].get());
     }
    }
template <typename Dtype>
void nTopVideoDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom,top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for(int i=0;i<prefetch_.size();++i){
      for(int j=0;j<prefetch_[i]->data_.size();++j){
          prefetch_[i]->data_[j]->mutable_cpu_data();
      }
      if(this->output_labels_){
          prefetch_[i]->label_.mutable_cpu_data();
      }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      for(int j=0;j<prefetch_[i]->data_.size();++j){
          prefetch_[i]->data_[j]->mutable_gpu_data();
      }
      if(this->output_labels_){
          prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#endif 
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";

}

template <typename Dtype>
void nTopVideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
      bottom, const vector<Blob<Dtype>*>& top){
  const int new_length = this->layer_param_.video_data_param().new_length();
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  const bool is_color  = this->layer_param_.video_data_param().is_color();
  string root_folder = this->layer_param_.video_data_param().root_folder();
  const int gap_example=this->layer_param_.video_data_param().gap_example();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
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
  bool read_video_result = ReadVideoToCVMat(root_folder +
                                            lines_[lines_id_].first,
                                            lines_[lines_id_].second,
                                            new_length, new_height, new_width,
                                            is_color,
                                            &cv_imgs,gap_example);
  CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                              " at frame " << lines_[lines_id_].second << ".";
  CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
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
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  int top_size=this->layer_param_.video_data_param().top_size();
  for (int i = 0; i < prefetch_.size(); ++i) {
      for(int j=0;j<prefetch_[i]->data_.size();++j){
        prefetch_[i]->data_[j].reset(new Blob<Dtype>());
        prefetch_[i]->data_[j]->Reshape(top_shape);
      }
  }
  
  for(int i=0;i<top_size;++i){
      top[i]->Reshape(top_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
      << top[0]->shape(1) << "," << top[0]->shape(2) << ","
      << top[0]->shape(3) << "," << top[0]->shape(4);
  // label
  vector<int> label_shape(1, batch_size);
  top[top_size]->Reshape(label_shape);
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void nTopVideoDataLayer<Dtype>::ShuffleVideos() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void nTopVideoDataLayer<Dtype>::load_batch(nTopBatch<Dtype>* batch){
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  int top_size=this->layer_param_.video_data_param().top_size();
  for(int i=0;i<batch->data_.size();++i){
      CHECK(batch->data_[i]->count());
  }
  CHECK(transformed_data_.count());
  CHECK_EQ(batch->data_.size(),top_size);
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_length = video_data_param.new_length();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  const bool is_color = video_data_param.is_color();
  string root_folder = video_data_param.root_folder(); 
  const int gap_example=this->layer_param_.video_data_param().gap_example();

  vector<Dtype*> prefetch_data(top_size);
  for(int i=0;i<top_size;++i){
      prefetch_data[i]=batch->data_[i]->mutable_cpu_data();
  }
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  const int lines_size=lines_.size();
  for(int item_id=0; item_id<batch_size;++item_id){
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    std::vector<vector<cv::Mat> > cv_imgs(top_size);
    for(int top_id=0;top_id<top_size;++top_id){
        bool read_video_result = ReadVideoToCVMat(root_folder +
                                               lines_[lines_id_].first,
                                               lines_[lines_id_].second+top_id*new_length,
                                               new_length, new_height,
                                               new_width, is_color, 
					       &cv_imgs[top_id],gap_example);
        CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                                    " at frame " << lines_[lines_id_].second+top_id*new_length << ".";
        CHECK_EQ(cv_imgs[top_id].size(), new_length) << "Could not load " <<
                                             lines_[lines_id_].first <<
                                            " at frame " <<
                                            lines_[lines_id_].second+top_id*new_length <<
                                            " correctly.";
        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset = batch->data_[top_id]->offset(item_id);
        transformed_data_.set_cpu_data(prefetch_data[top_id] + offset);
        const bool is_video = true;
        this->data_transformer_->Transform(cv_imgs[top_id], &(transformed_data_),
                                            is_video);
        trans_time += timer.MicroSeconds();
    }
    prefetch_label[item_id] = lines_[lines_id_].third;
    // go to the next iter
    ++lines_id_;
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
void nTopVideoDataLayer<Dtype>::InternalThreadEntry(){
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      nTopBatch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        for(int i=0;i<batch->data_.size();++i)  
            batch->data_[i]->data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
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
void nTopVideoDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  int top_size=this->layer_param_.video_data_param().top_size();
  CHECK_EQ(top_size,prefetch_current_->data_.size());
  for(int i=0;i<top_size;++i)
  {
    top[i]->ReshapeLike(*(prefetch_current_->data_[i]));
    top[i]->set_cpu_data(prefetch_current_->data_[i]->mutable_cpu_data());
  }
  
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[top_size]->ReshapeLike(prefetch_current_->label_);
    top[top_size]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
}

INSTANTIATE_CLASS(nTopVideoDataLayer);
REGISTER_LAYER_CLASS(nTopVideoData);

}//namespace caffe
#endif  //USE_OPENCV
