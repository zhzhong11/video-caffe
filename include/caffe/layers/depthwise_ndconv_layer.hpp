#ifndef CAFFE_DEPTHWISE_NDCONV_LAYER_HPP_
#define CAFFE_DEPTHWISE_NDCONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/cudnn_ndconv_layer.hpp"

namespace caffe {

    template <typename Dtype>
    class DepthwiseNdConvolutionLayer : public CudnnNdConvolutionLayer<Dtype> {
     public:
        explicit DepthwiseNdConvolutionLayer(const LayerParameter& param)
            : CudnnNdConvolutionLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      	    const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const{return "DepthwiseNdConvolution";}

     protected:
       // virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        //    const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
       // virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
       //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        //virtual inline bool reverse_dimensions() { return false; }
        //virtual void compute_output_shape();
    };
}   //namespace caffe

#endif
