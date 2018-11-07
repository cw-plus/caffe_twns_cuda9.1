#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/binary.hpp"

extern bool BINARY;
extern bool TERNARY;

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {  //计算卷积层的输出形状 
	const int* kernel_shape_data = this->kernel_shape_.cpu_data(); //卷积核大小   
	const int* stride_data = this->stride_.cpu_data();
	const int* pad_data = this->pad_.cpu_data();  // 
	const int* dilation_data = this->dilation_.cpu_data();  //卷积核膨胀  
	this->output_shape_.clear();
	for (int i = 0; i < this->num_spatial_axes_; ++i) {  // 空间轴个数 N*C*H*W
		// i + 1 to skip channel axis
		const int input_dim = this->input_shape(i + 1); //在这里获取输入blob的height与width 
		const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1; //在这里进行卷积核的扩展操作 
		const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
							   / stride_data[i] + 1;  //计算卷积过后生成的blob的高和宽 著名公式 
		this->output_shape_.push_back(output_dim);
	}
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	//const Dtype* weight = this->blobs_[0]->cpu_data(); //读入卷积层的参数（权重），blobs_[0]存储的权重，而blobs_[1]存储的偏置   
if(BINARY){
  this->blobs_[0]->binarize_data();
} 

if(TERNARY){
  this->blobs_[0]->ternarize_data(this->phase_); //是相当于保留两套权重? 看起来是分别放到
  // mutable_cpu_data 和 mutable_cpu_binary 里了
/*
    Dtype alpha = (Dtype) this->blobs_[0]->get_alpha();

for(int i=0; i<bottom.size(); i++){
  Blob<Dtype>* blob = bottom[i];
  caffe_cpu_scale(blob->count(), alpha, blob->cpu_data(), blob->mutable_cpu_data());
}
*/

}
  const Dtype* weight = (BINARY || TERNARY) ? this->blobs_[0]->cpu_binary() : this->blobs_[0]->cpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->cpu_data(); //读入bottom blob的data
		Dtype* top_data = top[i]->mutable_cpu_data();
		for (int n = 0; n < this->num_; ++n) {  //这里的num_指的是batch_size，也就是说，一张一张图片的来 	
			this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
								   top_data + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->cpu_data();
				this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
			}
		}
	}
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//const Dtype* weight = this->blobs_[0]->cpu_data(); //读入权重参数   
    const Dtype* weight = (BINARY || TERNARY) ? this->blobs_[0]->cpu_binary() : this->blobs_[0]->cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff(); //读入权重的梯度 
	for (int i = 0; i < top.size(); ++i) { //反向传播时 有类似于两个conv同时传入到一个conv层
		const Dtype* top_diff = top[i]->cpu_diff();  //获取每个top blob的梯度    
		const Dtype* bottom_data = bottom[i]->cpu_data();//获取每个bottom blob的数据  
		//已经算好了？   
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();//获取每个bottom blob的梯度 
		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) { //如果这个blob需要反传并且启用了偏置的话 
			Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff(); //获取该层偏置的梯度  
			for (int n = 0; n < this->num_; ++n) {  //对于每张输入的原图片偏置梯度的反传 
				this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
			}
		}
		if (this->param_propagate_down_[0] || propagate_down[i]) {
			for (int n = 0; n < this->num_; ++n) {
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {          //如果该blob需要反传权值梯度，则反传 
					this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
										  top_diff + n * this->top_dim_, weight_diff);
				}
				// gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) { //如果该blob需要反传数据梯度，则反传 	
					this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
											bottom_diff + n * this->bottom_dim_);
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
