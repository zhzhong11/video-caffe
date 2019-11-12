#include "caffe/layers/scalMulti_layer.hpp"

namespace caffe{
    
template<typename Dtype>
void ScalMultiLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
    CHECK_EQ(bottom[0]->shape(0),bottom[1]->shape(0));
    CHECK_EQ(bottom[0]->shape(1),bottom[1]->shape(1));
    if(bottom[0]->num_axes()==5){
        CHECK_EQ(bottom[0]->shape(2),bottom[1]->shape(2));
        CHECK_EQ(bottom[0]->shape(3),1);
        CHECK_EQ(bottom[0]->shape(4),1);
        spatialIndex=3;
    }else{
        CHECK_EQ(bottom[0]->shape(2),1);
        CHECK_EQ(bottom[0]->shape(3),1);
        spatialIndex=2;
    }
    top[0]->ReshapeLike(*bottom[1]);
    int spatial_dim=bottom[1]->count(spatialIndex);
    if(spatial_sum_multiplier_.count()<spatial_dim){
        spatial_sum_multiplier_.Reshape(vector<int>(1,spatial_dim));
        caffe_set(spatial_dim,Dtype(1),spatial_sum_multiplier_.mutable_cpu_data());
    }
}

template<typename Dtype>
void ScalMultiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    int channel_dim=bottom[1]->channels();
    int length_dim;
    if(bottom[1]->num_axes()==5)
        length_dim=bottom[1]->length();
    else
        length_dim=1;
    int spatial_dim=bottom[1]->count(spatialIndex);
    const Dtype* scale_data=bottom[0]->cpu_data();
    const Dtype* x_data=bottom[1]->cpu_data();
    Dtype* top_data=top[0]->mutable_cpu_data();
    //caffe_copy(bottom[1]->count(),bottom[1]->cpu_data(),top_data);
    for(int n=0;n<bottom[1]->num();++n){
        for(int c=0;c<channel_dim;++c){
            for(int l=0;l<length_dim;++l){
                int scale_offset=(n*channel_dim+c)*length_dim+l;
                caffe_cpu_scale(spatial_dim,scale_data[scale_offset],
               x_data+scale_offset*spatial_dim,top_data+scale_offset*spatial_dim);
            }
        }
    }
}

template<typename Dtype>
void ScalMultiLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    const int count=top[0]->count();
    const Dtype* top_diff=top[0]->cpu_diff();
    const Dtype* x_data=bottom[1]->cpu_data();
    const Dtype* scale_data=bottom[0]->cpu_data();
    int spatial_dim=bottom[1]->count(spatialIndex);
    if(propagate_down[0]){
        Dtype* x_diff=bottom[1]->mutable_cpu_diff();
        Dtype* scale_diff=bottom[0]->mutable_cpu_diff();
        caffe_mul(count,top_diff,x_data,x_diff);
        caffe_set(bottom[0]->count(),Dtype(0),scale_diff);
        caffe_cpu_gemv(CblasNoTrans, bottom[0]->count(), spatial_dim, Dtype(1),
        x_diff, spatial_sum_multiplier_.cpu_data(), Dtype(1), scale_diff); 
        if(!propagate_down[1]){
            caffe_set(bottom[1]->count(),Dtype(0),x_diff);
        }
    }
    if(propagate_down[1]){
        int channel_dim=bottom[1]->channels();
	int length_dim;
        if(bottom[1]->num_axes()==5)
            length_dim=bottom[1]->length();
        else
            length_dim=1;
        Dtype* x_diff=bottom[1]->mutable_cpu_diff();
        caffe_set(bottom[1]->count(),Dtype(0),x_diff);
        for(int n=0;n<bottom[1]->num();++n){
            for(int c=0;c<channel_dim;++c){
                for(int l=0;l<length_dim;++l){
                    int scale_offset=(n*channel_dim+c)*length_dim+l;
                    caffe_cpu_scale(spatial_dim,scale_data[scale_offset],
                    top_diff+scale_offset*spatial_dim,x_diff+scale_offset*spatial_dim);
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(ScalMultiLayer);
#endif

INSTANTIATE_CLASS(ScalMultiLayer);
REGISTER_LAYER_CLASS(ScalMulti);

} //namespace
