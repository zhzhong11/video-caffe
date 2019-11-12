#include <vector>
#include "caffe/layers/ntop_video_data_layer.hpp"

namespace caffe{

template <typename Dtype>
void nTopVideoDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
    top[i]->set_gpu_data(prefetch_current_->data_[i]->mutable_gpu_data());
  }
  
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[top_size]->ReshapeLike(prefetch_current_->label_);
    top[top_size]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(nTopVideoDataLayer);

}//namespace caffe
