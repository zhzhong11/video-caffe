#include "caffe/layers/continuation_indicator_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void ContinuationIndicatorLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
        ContinuationIndicatorParameter param = this->layer_param_.continuation_indicator_param();
        mini_batch_ = param.batch_size();
        time_step_ = param.time_step();
        CHECK_GT(mini_batch_, 0) << "The batch size should be greater than 0.";
        CHECK_GT(time_step_, 0) << "The time step should be greater than 0.";
    }
    template <typename Dtype>
    void ContinuationIndicatorLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape(2);
		top_shape[0] = time_step_;
		top_shape[1] = mini_batch_;
        top[0]->Reshape(top_shape); 
    }
    template <typename Dtype>
    void ContinuationIndicatorLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        CHECK_EQ(top[0]->shape()[0], time_step_) << "1st dimension of top blob should be same with time step.";
        CHECK_EQ(top[0]->shape()[1], mini_batch_) << "2nd dimension of top blob should be same with batch size.";
        Dtype* top_data = top[0]->mutable_cpu_data();
        vector<bool> flag(mini_batch_,this->layer_param_.continuation_indicator_param().flag());
        if(bottom.size()==1){
			for(int item_id=0; item_id< mini_batch_; ++item_id){
                if(abs(bottom[0]->cpu_data()[item_id]-1)<=0.01)
				    flag[item_id]=true;
            }	
		}
            
        for(int t = 0; t < time_step_; ++t) {
            for(int b = 0; b < mini_batch_; ++b) {
                // time step index: t, batch index: b
                if(flag[b])
                    *top_data++=Dtype(1);
                else
                    *top_data++ = t == 0? Dtype(0): Dtype(1);
            }
        }
    }
#ifdef CPU_ONLY
STUB_GPU(ContinuationIndicatorLayer);
#endif
INSTANTIATE_CLASS(ContinuationIndicatorLayer);
REGISTER_LAYER_CLASS(ContinuationIndicator);
}
