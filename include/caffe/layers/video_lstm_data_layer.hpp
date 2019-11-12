#ifndef CAFFE_VIDEO_LSTM_DATA_LAYER_HPP
#define CAFFE_VIDEO_LSTM_DATA_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe{

template <typename Dtype>
class LstmBatch {
    public:
    Blob<Dtype> data_, label_,lstm_flag_;
};

/**
 * @brief Provides data to the lstm Net from video files.
 */
template <typename Dtype>
class VideoLstmDataLayer :public BaseDataLayer<Dtype>, public InternalThread{
    public:
    explicit VideoLstmDataLayer(const LayerParameter& param);
    virtual ~VideoLstmDataLayer();

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

    virtual inline const char* type() const { return "VideoLstmData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
    virtual void InternalThreadEntry();
    shared_ptr<Caffe::RNG> prefetch_rng_;
    virtual void ShuffleVideos();
    virtual void load_batch(LstmBatch<Dtype>* batch);

    vector<shared_ptr<LstmBatch<Dtype> > > prefetch_;
    BlockingQueue<LstmBatch<Dtype>*> prefetch_free_;
    BlockingQueue<LstmBatch<Dtype>*> prefetch_full_;
    LstmBatch<Dtype>* prefetch_current_;

    Blob<Dtype> transformed_data_;

    vector<triplet> lines_;
    int batch_size_,new_length_,new_height_,new_width_,gap_example_;
    bool is_color_;
    int flag_new_item_; //记录上一次batch更新的lines_id_
    string root_folder_;
    vector<int> lines_id_;
    vector<int> current_video_id_;
    vector<int> cir_num_;
};

}//namespace caffe

#endif //CAFFE_VIDEO_LSTM_DATA_LAYER_HPP
