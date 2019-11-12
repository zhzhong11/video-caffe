#include <vector>
#include "caffe/layers/depthwise_ndconv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
#define CUDNN_STREAMS_PER_GROUP 3

template <typename Dtype>
void DepthwiseNdConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
          
  ConvolutionParameter conv_param =
    this->layer_param_.convolution_param();
  // Configure the kernel size, padding, stride, and inputs.
  CHECK(conv_param.has_kernel_shape())
      << "Kernel shape is required.";
  if (conv_param.has_pad_shape()) {
    CHECK_EQ(conv_param.kernel_shape().dim_size(),
             conv_param.pad_shape().dim_size())
        << "Kernel and Pad shape don't match !";
  }
  if (conv_param.has_stride_shape()) {
    CHECK_EQ(conv_param.kernel_shape().dim_size(),
             conv_param.stride_shape().dim_size())
        << "Kernel and Stride shape don't match !";
  }
  for (int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
    this->kernel_shape_.push_back(conv_param.kernel_shape().dim(i));
    CHECK_GT(this->kernel_shape_[i], 0) << "Filter dimensions cannot be zero.";
  }
  if (conv_param.has_pad_shape()) {
    for (int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
      this->pad_shape_.push_back(conv_param.pad_shape().dim(i));
    }
  } else {
    this->pad_shape_ = std::vector<int>(this->kernel_shape_.size(), 0);
  }
  if (conv_param.has_stride_shape()) {
    for (int i = 0; i < conv_param.kernel_shape().dim_size(); ++i) {
      this->stride_shape_.push_back(conv_param.stride_shape().dim(i));
    }
  } else {
    this->stride_shape_ = std::vector<int>(this->kernel_shape_.size(), 1);
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(1);
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_GT(this->num_output_, 0);
  CHECK_EQ(this->channels_, this->num_output_)
      << "Number of output should be equal to input.";

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();

  vector<int> weight_shape(this->kernel_shape_);
  weight_shape.insert(weight_shape.begin(), 1 );
  weight_shape.insert(weight_shape.begin(), this->num_output_);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      vector<int> bias_shape(1, this->num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Initialize CUDA streams and cuDNN.
  this->stream_ = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  this->handle_ = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  this->workspaceSizeInBytes = 0;
  this->workspace_data_ = NULL;

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&(this->stream_[g])));
    CUDNN_CHECK(cudnnCreate(&(this->handle_[g])));
    CUDNN_CHECK(cudnnSetStream(this->handle_[g], this->stream_[g]));
  }

  // Set the indexing parameters.
  weight_shape[0] /= this->group_;
  this->weight_offset_ = 1;
  for (int i = 0; i < weight_shape.size(); ++i) {
    this->weight_offset_ *= weight_shape[i];
  }
  this->bias_offset_ = weight_shape[0];

  // Create filter descriptor.
  cudnn::createNdFilterDesc<Dtype>(&(this->filter_desc_), weight_shape);

  this->bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  this->bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];
  this->workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  this->workspace_bwd_data_sizes_ = new size_t[bottom.size()];
  this->workspace_ = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];
  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensorDesc<Dtype>(&bottom_desc);
    this->bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensorDesc<Dtype>(&top_desc);
    this->top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    this->conv_descs_.push_back(conv_desc);
    this->workspace_bwd_data_sizes_[i] = 0;
    this->workspace_bwd_filter_sizes_[i] = 0;
    this->bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    this->bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensorDesc<Dtype>(&(this->bias_desc_));
  }

  this->handles_setup_ = true;

  

}

INSTANTIATE_CLASS(DepthwiseNdConvolutionLayer);
REGISTER_LAYER_CLASS(DepthwiseNdConvolution);
}  // namespace caffe
