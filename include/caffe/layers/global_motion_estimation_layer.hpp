#ifndef GLOBAL_MOTION_ESTIMATION_LAYER_HPP_
#define GLOBAL_MOTION_ESTIMATION_LAYER_HPP_

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



namespace caffe {

template<typename Dtype>
class globalBatch{
public:
  Blob<Dtype> data_, label_,globalParameter_;
};

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class GlobalMotionEstimationLayer : public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit GlobalMotionEstimationLayer(const LayerParameter& param);
  virtual ~GlobalMotionEstimationLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GlobalMotionEstimation"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void InternalThreadEntry();
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleVideos();
  virtual void load_batch(globalBatch<Dtype>* batch);

  vector<shared_ptr<globalBatch<Dtype> > > prefetch_;
  BlockingQueue<globalBatch<Dtype>*> prefetch_free_;
  BlockingQueue<globalBatch<Dtype>*> prefetch_full_;
  globalBatch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;
  Blob<Dtype> transformed_globalParameter_;

  vector<triplet> lines_;
  int lines_id_;
  vector<int>  ignore_example_list_;
  int batch_size_,new_length_,new_height_,new_width_,gap_example_;
  bool is_color_,is_flow_;
  string root_folder_;
};



}  // namespace caffe

#endif  // CAFFE_VIDEO_DATA_LAYER_HPP_
