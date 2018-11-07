#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/* 
输入层：（M_, N_, 1, 1); 
输出层： (M_, K_, 1, 1); 
W矩阵：（N_,K_,1,1); 
b矩阵：（N_,1,1,1); 
M_样本个数，K_单个样本特征长度，N_全连接之后神经元的个数。
*/  
template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //通过读取配置proto文件获得输出神经元的个数及是否使用偏置项
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  //全连接之后输出的神经元的个数
  N_ = num_output;
  
  //全连接层输出的Blob维数为 样本的个数*输出神经元的个数*1*1（M*N）  
  //这里axis=1，即从C开始展开，即，NCHW
  //输出：n_1 * (c_1 + c_2 + ... + c_K) * h * w	
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else { //如果配置文件使用偏置项，则开辟2个Blob类智能指针，否则开辟一个
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);//新开辟一个blob，指针返回给blobs_[0],weight_shape[2]为刚刚初始化的；
    if (transpose_) { 
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    //blobs_[0]指向权重矩阵，blobs_[1]指向偏置矩阵 ，全连接层，形状为N_*K_*1*1
    
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights //根据配置文件中的权重核（ weight_filler )的类型初始化填充权重矩阵blobs_[0];
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {//填充偏置矩阵blobs_[1]，每个输出单元对应一个偏置，共N_个  
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

//一批次处理多个样本，在每一批次中权重矩阵与偏置矩阵是不变的
template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

/* 
  计算y=W'*x + b, X表示输入，y表示输出  
  x为输入，维度 M_*K_    
  y为输出，维度 M_*N_    
  W为权重，维度 N_*K_, W_diff权重的梯度维度也为N_*K_  
  b为偏置，维度 N_*1_  
*/  
template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data(); //内存中的权重矩阵是N*K

/*
  caffe_cpu_gemm() 功能： C←αA×B+βC
  前两个参数控制A,B是否转置，其中A(bottom_data)维度是M_xK_,B(weight')维度是K_xN_，C(top_data)维度为M_xN_,
  最终 y = X*W', 维度为 M_xN_  */
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
	
/* 表示y= y + b (bias_multiplier维度为M_*1, b为1*N_(b实际上是N_*1，但是存储方式与1*N_等价，top_data为M_*N_)
实际是相当于将b复制成了M_*N_的矩阵，类似matlab的repmat(b, [M_, 1])，然后和top_data相加  */
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

	
	
//参考：http://www.caffecn.cn/?/question/36
template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //data传递的是数据，diff传递的是梯度，top_diff的维度是N*M，每一列代表一个样本的error term 
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    //A(top_diff'):N_*M_, B(bottom_data):M_*K_, C(W_diff)：N_*K_    
    //W_diff = top_diff' * bottom_data  
	// 这里为更新W，top_diff表示的是残差\delte,bottom_data表示的是上一层的激活值a
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    // top_diff(M_*N_), bias_multiplier(M_*1), b_diff(N_1)  
    // b_diff = top_diff' * bias_multiplier, 注意和gemm接口的区别  
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    // A(top_diff) M_*N_ , B(weight) N_*K_, C(bottom_diff) M_*K_  
    // bottom_diff = top_diff * weight 
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
