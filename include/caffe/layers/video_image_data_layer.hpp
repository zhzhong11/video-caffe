//
// Created by zhzhong on 19-11-6.
//

#ifndef CAFFE_VIDEO_IMAGE_DATA_LAYER_H
#define CAFFE_VIDEO_IMAGE_DATA_LAYER_H

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
    class ImageVideoDataBatch {
    public:
        Blob<Dtype> data_, label_,image_;
    };

/**
 * @brief Provides data to the 2d+3d Net from video files.
 */
    template <typename Dtype>
    class ImageVideoDataLayer :public BaseDataLayer<Dtype>, public InternalThread{
    public:
        explicit ImageVideoDataLayer(const LayerParameter& param);
        virtual ~ImageVideoDataLayer();

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

        virtual inline const char* type() const { return "ImageVideoData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 3; }

    protected:
        virtual void InternalThreadEntry();
        shared_ptr<Caffe::RNG> prefetch_rng_;
        virtual void ShuffleVideos();
        virtual void load_batch(ImageVideoDataBatch<Dtype>* batch);

        vector<shared_ptr<ImageVideoDataBatch<Dtype> > > prefetch_;
        BlockingQueue<ImageVideoDataBatch<Dtype>*> prefetch_free_;
        BlockingQueue<ImageVideoDataBatch<Dtype>*> prefetch_full_;
        ImageVideoDataBatch<Dtype>* prefetch_current_;

        Blob<Dtype> transformed_data_;
        Blob<Dtype> transformed_image_;

        vector<triplet> lines_;
        int batch_size_,new_length_,new_height_,new_width_,gap_example_;
        bool is_color_;
        int flag_new_item_; //?????batch???lines_id_
        string root_folder_;
        vector<int> lines_id_;
        vector<int> current_video_id_;
        vector<int> cir_num_;
    };

}//namespace caffe

#endif //CAFFE_VIDEO_IMAGE_DATA_LAYER_HPP
