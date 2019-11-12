#include "caffe/layers/scalMulti_layer.hpp"

namespace caffe{

template <typename Dtype>
__global__ void ScalMultiForward(const int nthreads, const int spatial_dim,
    const Dtype* scale_data, const Dtype* x_data, Dtype* top_data){
        CUDA_KERNEL_LOOP(index,nthreads){
            top_data[index]=scale_data[index/spatial_dim]*x_data[index];
        }
}

template <typename Dtype>
void ScalMultiLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    const Dtype* scale_data=bottom[0]->gpu_data();
    const Dtype* x_data=bottom[1]->gpu_data();
    Dtype* top_data=top[0]->mutable_gpu_data();
    const int count=bottom[1]->count();
    ScalMultiForward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[1]->count(spatialIndex), scale_data, x_data, top_data);
}

template<typename Dtype>
__global__ void ScalMultiBackwardScale(const int nthreads, const int spatial_dim,
    const Dtype* x_data, const Dtype* top_diff, Dtype* sacle_diff){
        __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
        unsigned int tid=threadIdx.x;
        buffer[tid]=0;
        __syncthreads();

        for(int j=tid;j<spatial_dim;j+=blockDim.x){
            int offset=blockIdx.x*spatial_dim+j;
            buffer[tid]+=top_diff[offset]*x_data[offset];
        }
        __syncthreads();

        for(int i=blockDim.x/2;i>0;i>>=1){
            if(tid<i){
                buffer[tid]+=buffer[tid+i];
            }
        }
        __syncthreads();
         if(tid==0){
            sacle_diff[blockIdx.x]=buffer[0];
        }
}

template <typename Dtype>
__global__ void  ScalMultiBackwardX(const int nthreads, const int spatial_dim,
    const Dtype* scale_data, const Dtype* top_diff, Dtype* x_diff){
    CUDA_KERNEL_LOOP(index, nthreads){
        x_diff[index]=scale_data[index/spatial_dim]*top_diff[index];
    }
}

template <typename Dtype>
void ScalMultiLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    const int count=top[0]->count();
    const Dtype* top_diff=top[0]->gpu_diff();
    if(propagate_down[0]){
        int nthreads=bottom[1]->count(0,spatialIndex);
        ScalMultiBackwardScale<Dtype><<<nthreads,CAFFE_CUDA_NUM_THREADS>>>(
            nthreads,bottom[1]->count(spatialIndex),bottom[1]->gpu_data(),top_diff,
            bottom[0]->mutable_gpu_diff());
    }
    if(propagate_down[1]){
        ScalMultiBackwardX<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
            count,top[0]->count(spatialIndex),bottom[0]->gpu_data(),top_diff,
            bottom[1]->mutable_gpu_diff());
    }
    CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_GPU_FUNCS(ScalMultiLayer);

} //namespace caffe
