#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template <typename Dtype>
    ImageVideoDataLayer<Dtype>::~ImageVideoDataLayer() {
        this->StopInternalThread();
    }

    template <typename Dtype>
    ImageVideoDataLayer<Dtype>::ImageVideoDataLayer(const LayerParameter& param)
            : BaseDataLayer<Dtype>(param),prefetch_(param.data_param().prefetch()),
              prefetch_free_(), prefetch_full_(), prefetch_current_(){
        for(int i=0;i<prefetch_.size();++i){
            prefetch_[i].reset(new ImageVideoDataBatch<Dtype>());
            prefetch_free_.push(prefetch_[i].get());
        }
    }
    template <typename Dtype>
    void ImageVideoDataLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BaseDataLayer<Dtype>::LayerSetUp(bottom,top);

        // Before starting the prefetch thread, we make cpu_data and gpu_data
        // calls so that the prefetch thread does not accidentally make simultaneous
        // cudaMalloc calls when the main thread is running. In some GPUs this
        // seems to cause failures if we do not so.
        for(int i=0;i<prefetch_.size();++i){
            prefetch_[i]->data_.mutable_cpu_data();
            prefetch_[i]->image_.mutable_cpu_data();
            if(this->output_labels_){
                prefetch_[i]->label_.mutable_cpu_data();
            }
        }
#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            for (int i = 0; i < prefetch_.size(); ++i) {
                prefetch_[i]->data_.mutable_gpu_data();
                prefetch_[i]->image_.mutable_gpu_data();
                if(this->output_labels_){
                    prefetch_[i]->label_.mutable_gpu_data();
                }
            }
        }
#endif
        DLOG(INFO) << "Initializing prefetch";
        this->data_transformer_->InitRand();
        this->image_transformer_->InitRand();
        StartInternalThread();
        DLOG(INFO) << "Prefetch initialized.";

    }
    template <typename Dtype>
    void ImageVideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
    bottom, const vector<Blob<Dtype>*>& top) {
        VideoDataParameter video_data_param=this->layer_param_.video_data_param();
        batch_size_ = video_data_param.batch_size();
        new_length_ = video_data_param.new_length();
        new_height_ = video_data_param.new_height();
        new_width_  = video_data_param.new_width();
        is_color_  = video_data_param.is_color();
        root_folder_ = video_data_param.root_folder();
        gap_example_ = video_data_param.gap_example();

        CHECK((new_height_ == 0 && new_width_ == 0) ||
              (new_height_ > 0 && new_width_ > 0)) << "Current implementation requires "
                                                      "new_height_ and new_width_ to be set at the same time.";
        CHECK_GT(batch_size_,0)<<"batch size must be great 0.";
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

        lines_id_.clear();
        current_video_id_.clear();
        cir_num_.clear();
        flag_new_item_ = batch_size_ - 1;
        for(int item_id = 0; item_id < batch_size_; ++item_id){
            lines_id_.push_back(item_id);
            current_video_id_.push_back(0);
            //Calculate the number of loops required to compute current item's video
            const int num_images=numFiles(root_folder_ + lines_[item_id].first);
            cir_num_.push_back(num_images/new_length_);
        }

        CHECK_EQ(lines_id_.size(),batch_size_);
        CHECK_EQ(current_video_id_.size(),batch_size_);
        CHECK_EQ(cir_num_.size(),batch_size_);

        int gap_example = gap_example_;
        // Check if we would need to randomly skip a few data points
        if (this->layer_param_.video_data_param().rand_skip()) {
            unsigned int skip = caffe_rng_rand() %
                                this->layer_param_.video_data_param().rand_skip();
            LOG(INFO) << "Skipping first " << skip << " data points.";
            CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
            for(int i=0;i<batch_size_;++i){
                lines_id_[i]+=skip;
            }
        }
        // Read a video clip, and use it to initialize the top blob.
        std::vector<cv::Mat> cv_imgs;
        bool read_video_result = ReadVideoToCVMat(root_folder_ +
                                                  lines_[lines_id_[0]].first,
                                                  1,
                                                  new_length_, new_height_, new_width_,
                                                  is_color_,
                                                  &cv_imgs,gap_example);
        CHECK(read_video_result) << "Could not load " << lines_[lines_id_[0]].first <<
                                 " at frame " << 1 << ".";
        CHECK_EQ(cv_imgs.size(), new_length_) << "Could not load " <<
                                              lines_[lines_id_[0]].first <<
                                              " at frame " <<
                                              1 <<
                                              " correctly.";

        // Use data_transformer to infer the expected blob shape from a cv_image.
        const bool is_video = true;
        vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                        is_video);
        this->transformed_data_.Reshape(top_shape);
        top_shape[0] = batch_size_;
        vector<int> image_shape = top_shape;
        image_shape.erase(image_shape.begin() + 2,image_shape.begin() + 3);
        this->transformed_image_.Reshape(image_shape);
        for (int i = 0; i < prefetch_.size(); ++i) {
            prefetch_[i]->data_.Reshape(top_shape);
            prefetch_[i]->image_.Reshape(image_shape);
        }

        //top[0] is video data, top[1] is image data, top[2] is labels
        top[0]->Reshape(top_shape);
        top[1]->Reshape(image_shape);
        LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
                  << top[0]->shape(1) << "," << top[0]->shape(2) << ","
                  << top[0]->shape(3) << "," << top[0]->shape(4);
        LOG(INFO) << "output image size: " << top[1]->shape(0) << ","
                  << top[1]->shape(1) << "," << top[1]->shape(2) << ","
                  << top[1]->shape(3);
        // label
        vector<int> label_shape(1, batch_size_);
        top[2]->Reshape(label_shape);
        for (int i = 0; i < prefetch_.size(); ++i) {
            prefetch_[i]->label_.Reshape(label_shape);
        }
    }
    template <typename Dtype>
    void ImageVideoDataLayer<Dtype>::ShuffleVideos() {
        caffe::rng_t* prefetch_rng =
                static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

    template <typename Dtype>
    void ImageVideoDataLayer<Dtype>::load_batch(ImageVideoDataBatch<Dtype>* batch){
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        int gap_example = gap_example_;
        CPUTimer timer;
        CHECK(batch->data_.count());
        CHECK(this->transformed_data_.count());
        CHECK(this->transformed_image_.count());

        Dtype* prefetch_data = batch->data_.mutable_cpu_data();
        Dtype* prefetch_label = batch->label_.mutable_cpu_data();
        Dtype* prefetch_image = batch->image_.mutable_cpu_data();
        // datum scales
        const int lines_size = lines_.size();
        for(int item_id = 0; item_id < batch_size_; ++item_id){
            //get a blob
            timer.Start();
            CHECK_GT(lines_size, lines_id_[item_id]);

            std::vector<cv::Mat> cv_imgs;
            cv::Mat images;
            bool read_video_result=ReadVideoToCVMat(root_folder_ +
                                                    lines_[lines_id_[item_id]].first,
                                                    current_video_id_[item_id]*new_length_+1,
                                                    new_length_, new_height_, new_width_,
                                                    is_color_, &cv_imgs, gap_example);
            CHECK(read_video_result) << "Could not load " << lines_[lines_id_[item_id]].first <<
                                     " at frame " << current_video_id_[item_id]*new_length_+1<< ".";
            read_time += timer.MicroSeconds();

            //取第20帧作为2D卷积的输入
            cv::Mat cv_img = cv_imgs[20];

            timer.Start();
            // Apply transformations (mirror, crop...) to the image
            int offset = batch->data_.offset(item_id);
            int offset1 = batch->image_.offset(item_id);
            this->transformed_data_.set_cpu_data(prefetch_data + offset);
            this->transformed_image_.set_cpu_data(prefetch_image + offset1);
            const bool is_video = true;
            this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
                                               is_video);
            this->image_transformer_->Transform(cv_img, &(this->transformed_image_));
            trans_time += timer.MicroSeconds();

            prefetch_label[item_id] = lines_[lines_id_[item_id]].third;
//            if(current_video_id_[item_id]==0)
//                prefetch_lstm_flag[item_id]=0;
//            else
//                prefetch_lstm_flag[item_id]=1;

            //go to the next iter
            ++current_video_id_[item_id];
            if(current_video_id_[item_id] >= cir_num_[item_id]){
                current_video_id_[item_id] = 0;
                lines_id_[item_id] = lines_id_[flag_new_item_] + 1;
                flag_new_item_ = item_id;

                if(lines_id_[item_id] >= lines_size){
                    DLOG(INFO) << "Restarting data prefetching from start.";
                    lines_id_[item_id]=0;
                }

                //Calculate the number of loops required to compute current item's video
                const int num_images=numFiles(root_folder_+lines_[lines_id_[item_id]].first);
                cir_num_[item_id]=num_images/new_length_;
            }
        }

        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    template <typename Dtype>
    void ImageVideoDataLayer<Dtype>::InternalThreadEntry(){
#ifndef CPU_ONLY
        cudaStream_t stream;
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
#endif

        try {
            while (!must_stop()) {
                ImageVideoDataBatch<Dtype>* batch = prefetch_free_.pop();
                load_batch(batch);
#ifndef CPU_ONLY
                if (Caffe::mode() == Caffe::GPU) {
                    batch->data_.data().get()->async_gpu_push(stream);
                    batch->image_.data().get()->async_gpu_push(stream);
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
    void ImageVideoDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top){
        if (prefetch_current_) {
            prefetch_free_.push(prefetch_current_);
        }
        prefetch_current_ = prefetch_full_.pop("Waiting for data");
        // Reshape to loaded data.
        top[0]->ReshapeLike(prefetch_current_->data_);
        top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
        top[1]->ReshapeLike(prefetch_current_->image_);
        top[1]->set_cpu_data(prefetch_current_->image_.mutable_cpu_data());
        if (this->output_labels_) {
            // Reshape to loaded labels.
            top[2]->ReshapeLike(prefetch_current_->label_);
            top[2]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
        }
    }

    INSTANTIATE_CLASS(ImageVideoDataLayer);
    REGISTER_LAYER_CLASS(ImageVideoData);
}// namespace caffe
#endif //USE_OPENCV
