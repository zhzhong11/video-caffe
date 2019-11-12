#include <vector>
#include <algorithm>
#include <cfloat>
#include "caffe/layers/depthwise_ndconv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template<typename Dtype>
    __global__ void ConvForward(const int nthreads,
                const Dtype* const bottom_data, const int num, const int channels,
                const int lenght, const int height, const int width, const int conved_lenght,
                const int conved_height, const int conved_width, const int kernel_l, const int kernel_h,
                const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                const int pad_l, const int pad_h, const int pad_w, Dtype * const top_data, const Dtype* const weight,
                const Dtype* const bias, const bool bias_term_){
                    CUDA_KERNEL_LOOP(index, nthreads) {

                        const int pw=index % conved_width;
                        const int ph=(index / conved_width) % conved_height;
                        const int pl=(index / conved_width / conved_height) % conved_lenght;
                        const int c=(index / conved_width / conved_height / conved_lenght) % channels;
                        const int n=index / conved_width /conved_height / conved_lenght /channels;

                        int lstart=pl*stride_l-pad_l;
                        int hstart=ph*stride_h-pad_h;
                        int wstart=pw*stride_w-pad_w;
                        int lend=min(lstart+kernel_l, lenght);
                        int hend=min(hstart+kernel_h, height);
                        int wend=min(wstart+kernel_w, width);

                        lstart=max(lstart,0);
                        hstart=max(hstart,0);
                        wstart=max(wstart,0);

                        Dtype aveval=0;
                        const Dtype* const bottom_slice=bottom_data+(n*channels+c)*lenght*height*width;
                        const Dtype* const weight_slice=weight+c*kernel_l*kernel_h*kernel_w;

                        int klstart=lend<kernel_l?kernel_l-lend:0;
                        int khstart=hend<kernel_h?kernel_h-hend:0;
                        int kwstart=wend<kernel_w?kernel_w-wend:0;

                        for(int l=lstart;l<lend;++l){
                            for(int h=hstart;h<hend;++h){
                                for(int w=wstart;w<wend;++w){
                                    aveval+=bottom_slice[(l*height+h)*width+w]*weight_slice[((klstart+l-lstart)*kernel_h+(khstart+h-hstart))*kernel_w+(kwstart+w-wstart)];
                                }
                            }
                        }
                        if(bias_term_){
                            aveval+=bias[c];
                        }
                        top_data[index]=aveval;
                    }
                }

    template<typename Dtype>
    void DepthwiseNdConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        const Dtype* weight = this->blobs_[0]->gpu_data();

        for(int i=0;i<bottom.size();++i){
            const Dtype* bottom_data = bottom[i]->gpu_data();
            Dtype* top_data = top[i]->mutable_gpu_data();
            const int count = top[i]->count();
            vector<int> shape_=bottom[i]->shape();
            const int channels_=shape_[1];
            const int lenght_=shape_[2];
            const int height_=shape_[3];
            const int width_=shape_[4];

            const int kernel_l_=this->kernel_shape_[0];
            const int kernel_h_=this->kernel_shape_[1];
            const int kernel_w_=this->kernel_shape_[2];
            const int stride_l_=this->stride_shape_[0];
            const int stride_h_=this->stride_shape_[1];
            const int stride_w_=this->stride_shape_[2];
            const int pad_l_=this->pad_shape_[0];
            const int pad_h_=this->pad_shape_[1];
            const int pad_w_=this->pad_shape_[2];

            const int conved_lenght=this->output_shape_[2];
            const int conved_height=this->output_shape_[3];
            const int conved_width=this->output_shape_[4];

            const bool bias_term_=this->bias_term_;

            if(bias_term_){
                const Dtype* const bias=this->blobs_[1]->gpu_data();
                ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, bottom[i]->num(), channels_, lenght_, height_, width_, conved_lenght, conved_height,
                    conved_width, kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_,
                    top_data, weight, bias, bias_term_);
            }else{
                ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, bottom[i]->num(), channels_, lenght_, height_, width_, conved_lenght, conved_height,
                    conved_width, kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_,
                    top_data, weight, 0, bias_term_);
            }
        }
    }

    template <typename Dtype>
    __global__ void ConvBackward(const int nthreads, const Dtype* const top_diff, const int num, const int channels,
                const int lenght, const int height, const int width, const int conved_lenght,
                const int conved_height, const int conved_width, const int kernel_l, const int kernel_h,
                const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                const int pad_l, const int pad_h, const int pad_w, Dtype* const bottom_diff, const Dtype* const weight){
                    CUDA_KERNEL_LOOP(index,nthreads){
                        const int w=index % width+pad_w;
                        const int h=(index / width) % height+pad_h;
                        const int l=(index / width / height) % lenght+pad_l;
                        const int c=(index / width / height / lenght) % channels;
                        const int n=index / width / height /lenght / channels;

                        const int plstart=(l<kernel_l)?0:(l-kernel_l)/stride_l+1;
                        const int plend=min(l/stride_l+1,conved_lenght);
                        const int phstart=(h<kernel_h)?0:(h-kernel_h)/stride_h+1;
                        const int phend=min(h/stride_h+1,conved_height);
                        const int pwstart=(w<kernel_w)?0:(w-kernel_w)/stride_w+1;
                        const int pwend=min(w/stride_w+1,conved_width);

                        const int klstart=(l>=kernel_l)?((l-kernel_l)%stride_l)+(kernel_l-stride_l):l;
                        const int khstart=(h>=kernel_h)?((h-kernel_h)%stride_h)+(kernel_h-stride_h):h;
                        const int kwstart=(w>=kernel_w)?((w-kernel_w)%stride_w)+(kernel_w-stride_w):w;

                        Dtype gradient=0;
                        const Dtype* const top_diff_slice=top_diff+(n*channels+c)*conved_lenght*conved_height*conved_width;
                        const Dtype* const weight_slice=weight+c*kernel_l*kernel_h*kernel_w;

                        for(int pl=plstart;pl<plend;++pl){
                            for(int ph=phstart;ph<phend;++ph){
                                for(int pw=pwstart;pw<pwend;++pw){
                                    int kl=klstart-(pl-plstart)*stride_l;
                                    int kh=khstart-(ph-phstart)*stride_h;
                                    int kw=kwstart-(pw-pwstart)*stride_w;
                                    gradient+=top_diff_slice[(pl*conved_height+ph)*conved_width+pw]*weight_slice[((kl*kernel_h+kh)*kernel_w+kw)];
                                }
                            }
                        }
                        bottom_diff[index]=gradient;

                    }
                }

    __device__ float atomicAddme(float* address, float val)
    {
        return atomicAdd(address,val);
    }

    __device__ double atomicAddme(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, 
                            __double_as_longlong(val + 
                            __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }


    #define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)

    template <typename Dtype>
    __global__ void ConvBackwardWeight(const int nthreads, const Dtype* const top_diff, const int num, const int channels,
                const int lenght, const int height, const int width, const int conved_lenght,
                const int conved_height, const int conved_width, const int kernel_l, const int kernel_h,
                const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                const int pad_l, const int pad_h, const int pad_w,Dtype* const weight_diff,const Dtype* const bottom_data){
                    CUDA_KERNEL_LOOP(index,nthreads){
                        const int kw=index % kernel_w;
                        const int kh=(index / kernel_w) % kernel_h;
                        const int kl=(index / kernel_w / kernel_h) % kernel_l;
                        const int c =index/ kernel_w / kernel_h / kernel_l;

                        Dtype gradient=0;
                        for(int n=0;n<num;++n){
                            const Dtype* const top_diff_slice=top_diff+(n*channels+c)*conved_lenght*conved_height*conved_width;
                            const Dtype* const bottom_data_slice=bottom_data+(n*channels+c)*lenght*height*width;

                            const int plstart=max(DIVIDE_CEIL((pad_l-kl),stride_l),0);
                            const int plend=min(DIVIDE_CEIL((lenght+pad_l-kl),stride_l),conved_lenght);
                            const int phstart=max(DIVIDE_CEIL((pad_h-kh),stride_h),0);
			                const int phend=min(DIVIDE_CEIL((height+pad_h-kh),stride_h),conved_height);
			                const int pwstart=max(DIVIDE_CEIL((pad_w-kw),stride_w),0);			
			                const int pwend=min(DIVIDE_CEIL((width+pad_w-kw),stride_w),conved_width);

                            for(int pl=plstart;pl<plend;++pl){
                                for(int ph=phstart;ph<phend;++ph){
                                    for(int pw=pwstart;pw<pwend;++pw){
                                        const int l=pl*stride_l+kl-pad_l;
                                        const int h=ph*stride_h+kh-pad_h;
					                    const int w=pw*stride_w+kw-pad_w;

                                        gradient+=top_diff_slice[(pl*conved_height+ph)*conved_width+pw]*bottom_data_slice[(l*height+h)*width+w];
                                    }
                                }
                            }
                        }
                        weight_diff[c*kernel_l*kernel_h*kernel_w+(kl*kernel_h+kh)*kernel_w+kw]+=gradient;
                    }
                }

    template <typename Dtype>
    __global__ void ConvBackwardBias(const int nthreads, const Dtype* const top_diff, const int num, const int channels,
                const int lenght, const int height, const int width, const int conved_lenght,
                const int conved_height, const int conved_width, const int kernel_l, const int kernel_h,
                const int kernel_w, const int stride_l, const int stride_h, const int stride_w,
                const int pad_l, const int pad_h, const int pad_w, Dtype* const bias_diff){
                    CUDA_KERNEL_LOOP(index,nthreads){
                        const int c=index;
                        Dtype gradient=0;
                        for(int n=0;n<num;n++){
                            const Dtype* const top_diff_slice=top_diff+(n*channels+c)*conved_lenght*conved_height*conved_width;

                            for(int pl=0;pl<conved_lenght;++pl){
                                for(int ph=0;ph<conved_height;++ph){
                                    for(int pw=0;pw<conved_width;++pw){
                                        gradient+=top_diff_slice[(pl*conved_height+ph)*conved_width+pw];
                                    }
                                }
                            }
                        }
                        bias_diff[c]+=gradient;
                    }
                }

    template<typename Dtype>
    void DepthwiseNdConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            const Dtype* weight=this->blobs_[0]->gpu_data();
            Dtype* weight_diff=this->blobs_[0]->mutable_gpu_diff();
            const bool bias_term_ = this->bias_term_;
	        Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
            const bool bias_propagate_down_ = this->param_propagate_down_[1];
	        const bool weight_propagate_down_ = this->param_propagate_down_[0];

            const int kernel_l_=this->kernel_shape_[0];
            const int kernel_h_=this->kernel_shape_[1];
            const int kernel_w_=this->kernel_shape_[2];
            const int stride_l_=this->stride_shape_[0];
            const int stride_h_=this->stride_shape_[1];
            const int stride_w_=this->stride_shape_[2];
            const int pad_l_=this->pad_shape_[0];
            const int pad_h_=this->pad_shape_[1];
            const int pad_w_=this->pad_shape_[2];

            const int conved_lenght=this->output_shape_[2];
            const int conved_height=this->output_shape_[3];
            const int conved_width=this->output_shape_[4];

            for(int i=0;i<top.size();++i){

                const Dtype* top_diff = top[i]->gpu_diff();
		        const Dtype* bottom_data = bottom[i]->gpu_data();
		        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

                vector<int> shape_=bottom[i]->shape();
                const int channels_=shape_[1];
                const int lenght_=shape_[2];
                const int height_=shape_[3];
                const int width_=shape_[4];

                //Bias gradient,if necessary.
                if(bias_term_&&bias_propagate_down_){
                    const int count_bias=channels_;
                    ConvBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(count_bias), CAFFE_CUDA_NUM_THREADS>>>(
                        count_bias, top_diff, bottom[i]->num(), channels_, lenght_, height_, width_, conved_lenght, conved_height, 
                        conved_width, kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_,
                        bias_diff);
                }
                // gradient w.r.t. weight. Note that we will accumulate diffs.
                if(weight_propagate_down_){
                    const int count_weight=channels_*kernel_l_*kernel_h_*kernel_w_;
                    ConvBackwardWeight<Dtype><<<CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS>>>(
                        count_weight, top_diff, bottom[i]->num(), channels_, lenght_, height_, width_, conved_lenght, conved_height,
                        conved_width, kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_,
                        weight_diff, bottom_data);
                }
                // gradient w.r.t. bottom data, if necessary.
                if(propagate_down[i]){
                    const int count_bottom=bottom[i]->count();
                    ConvBackward<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
                        count_bottom, top_diff, bottom[i]->num(), channels_, lenght_, height_, width_, conved_lenght, conved_height,
                        conved_width, kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_,
                        bottom_diff, weight);
                }
            }
        }

    INSTANTIATE_LAYER_GPU_FUNCS (DepthwiseNdConvolutionLayer);


}//namespace caffe
