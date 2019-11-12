#include <vector>
#include "caffe/layers/global_motion_estimation_layer.hpp"

namespace caffe{
    template <typename Dtype>
    void GlobalMotionEstimationLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      if (prefetch_current_) {
        prefetch_free_.push(prefetch_current_);
      }
      prefetch_current_ = prefetch_full_.pop("Waiting for data");
      // Reshape to loaded data.
      top[0]->ReshapeLike(prefetch_current_->data_);
      top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
      if (this->output_labels_) {
        // Reshape to loaded labels.
        top[1]->ReshapeLike(prefetch_current_->label_);
        top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
      }
      top[2]->ReshapeLike(prefetch_current_->globalParameter_);
      top[2]->set_gpu_data(prefetch_current_->globalParameter_.mutable_gpu_data());
	  
    }
    
    INSTANTIATE_LAYER_GPU_FORWARD(GlobalMotionEstimationLayer);

}//namespace caffe
