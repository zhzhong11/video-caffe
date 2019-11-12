#ifndef CAFFE_NTOP_VIDEO_DATA_LAYER_HPP_
#define CAFFE_NTOP_VIDEO_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"

// an extension the std::pair which used to store image filename and
// its label (int). now, a frame number associated with the video filename
// is needed (second param) to fully represent a video segment
namespace caffe{

template <typename Dtype>
class nTopBatch {
    public:
    nTopBatch(int topsize):data_(topsize){}
    vector<shared_ptr<Blob<Dtype> > > data_;
    Blob<Dtype> label_;
};

template <typename Dtype>
class nTopVideoDataLayer :public BaseDataLayer<Dtype>, public InternalThread {
    public:
    explicit nTopVideoDataLayer(const LayerParameter& param);
    virtual ~nTopVideoDataLayer();
      // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
    void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const{ return "nTopVideoData";}
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int MinTopBlobs() const { return 2; }


    protected:
    virtual void InternalThreadEntry();
    shared_ptr<Caffe::RNG> prefetch_rng_;
    virtual void ShuffleVideos();
    virtual void load_batch(nTopBatch<Dtype>* batch);

    vector<shared_ptr<nTopBatch<Dtype> > > prefetch_;
    BlockingQueue<nTopBatch<Dtype>*> prefetch_free_;
    BlockingQueue<nTopBatch<Dtype>*> prefetch_full_;
    nTopBatch<Dtype>* prefetch_current_;

    Blob<Dtype> transformed_data_;
    vector<triplet> lines_;
    int lines_id_;

};

}
#endif
