/*
 * cnn.h
 *
 *  Created on: Apr 27, 2017
 *      Author: copper
 */

#ifndef CNN_H_
#define CNN_H_

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#define FORWARD_GPU
#define BACKWARD_GPU
// #define UPDATE_GPU

#ifdef FORWARD_GPU
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#define GPU_OPT_LOCAL_MEM
#endif

#ifdef FORWARD_GPU
std::string status2str(cl_int status);

#define CHECK_CL(status, str)                                                  \
  if (status != CL_SUCCESS) {                                                  \
    std::cout << "OpenCL Failure: " << str << "\n" << status2str(status) << "\n"       \
              << std::flush;                                                   \
    exit(1);                                                                   \
  }

std::string get_device_info(cl_device_id device, cl_device_info params);
#endif

// 各层图像大小
#define width_image_input_CNN 32  //归一化图像宽
#define height_image_input_CNN 32 //归一化图像高
#define width_image_C1_CNN 28
#define height_image_C1_CNN 28
#define width_image_S2_CNN 14
#define height_image_S2_CNN 14
#define width_image_C3_CNN 10
#define height_image_C3_CNN 10
#define width_image_S4_CNN 5
#define height_image_S4_CNN 5
#define width_image_C5_CNN 1
#define height_image_C5_CNN 1
#define width_image_output_CNN 1
#define height_image_output_CNN 1

// 卷积核大小
#define width_kernel_conv_CNN 5 //卷积核大小
#define height_kernel_conv_CNN 5
#define width_kernel_pooling_CNN 2
#define height_kernel_pooling_CNN 2
#define size_pooling_CNN 2

// 特征图数量   feature maps
#define num_map_input_CNN 1   //输入层map个数
#define num_map_C1_CNN 6      // C1层map个数
#define num_map_S2_CNN 6      // S2层map个数
#define num_map_C3_CNN 16     // C3层map个数
#define num_map_S4_CNN 16     // S4层map个数
#define num_map_C5_CNN 120    // C5层map个数
#define num_map_output_CNN 10 //输出层map个数

// MNIST
#define num_patterns_train_CNN 60000 // 60000 //训练模式对数(总数)
#define num_patterns_test_CNN 10000  // 10000 //测试模式对数(总数)

// Train
#define num_epochs_CNN 1e100      //最大迭代次数
#define accuracy_rate_CNN 0.985 //要求达到的准确率
#define learning_rate_CNN 0.01  //学习率
#define eps_CNN 1e-8

//
#define len_weight_C1_CNN 150      // C1层权值数，5*5*6*1=150
#define len_bias_C1_CNN 6          // C1层阈值数，6
#define len_weight_S2_CNN 6        // S2层权值数,1*6=6
#define len_bias_S2_CNN 6          // S2层阈值数,6
#define len_weight_C3_CNN 2400     // C3层权值数，5*5*16*6=2400
#define len_bias_C3_CNN 16         // C3层阈值数,16
#define len_weight_S4_CNN 16       // S4层权值数，1*16=16
#define len_bias_S4_CNN 16         // S4层阈值数，16
#define len_weight_C5_CNN 48000    // C5层权值数，5*5*16*120=48000
#define len_bias_C5_CNN 120        // C5层阈值数，120
#define len_weight_output_CNN 1200 //输出层权值数，120*10=1200
#define len_bias_output_CNN 10     //输出层阈值数，10

#define num_neuron_input_CNN 1024 //输入层神经元数，32*32=1024
#define num_neuron_C1_CNN 4704    // C1层神经元数，28*28*6=4704
#define num_neuron_S2_CNN 1176    // S2层神经元数，14*14*6=1176
#define num_neuron_C3_CNN 1600    // C3层神经元数，10*10*16=1600
#define num_neuron_S4_CNN 400     // S4层神经元数，5*5*16=400
#define num_neuron_C5_CNN 120     // C5层神经元数，1*120=120
#define num_neuron_output_CNN 10  //输出层神经元数，1*10=10

//
class CNN {
public:
#ifdef FORWARD_GPU
  CNN(int platform_id = 0, int device_id = 0);
#else
  CNN();
#endif
  ~CNN();

  void init();
  bool train();
  int predict(const unsigned char *data, int width, int height);
  bool readModelFile(const char *name);
  bool saveMiddlePic(int index);

protected:
  void release(); //释放申请的空间

  // init
  bool initWeightThreshold(); //初始化，产生[-1, 1]之间的随机小数
  void init_variable(float *val, float c, int len);
  bool uniform_rand(float *src, int len, float min, float max);
  float uniform_rand(float min, float max);

  // mnist
  int reverseInt(int i);
  void readMnistImages(std::string filename, float *data_dst, int num_image);
  void readMnistLabels(std::string filename, float *data_dst, int num_image);
  bool getSrcData(); //读取MNIST数据

  // math_functions
  float activation_function_tanh(float x);            //激活函数:tanh
  float activation_function_tanh_derivative(float x); //激活函数tanh的导数
  float activation_function_identity(float x);
  float activation_function_identity_derivative(float x);
  float loss_function_mse(float y, float t); //损失函数:mean squared error
  float loss_function_mse_derivative(float y, float t);
  void loss_function_gradient(const float *y, const float *t, float *dst,
                              int len);
  float dot_product(const float *s1, const float *s2, int len); //点乘
  bool muladd(const float *src, float c, int len,
              float *dst); // dst[i] += c * src[i]

  // model
  // bool reaModelFile(const char *name);
  bool saveModelFile(
      const char *name); //将训练好的model保存起来，包括各层的节点数，权值和阈值

  //
  int get_index(int x, int y, int channel, int width, int height, int depth);

  bool Forward_C1(bool train=true); //前向传播
  bool Forward_S2();
  bool Forward_C3();
  bool Forward_S4();
  bool Forward_C5();
  bool Forward_output();

  bool Backward_output();
  bool Backward_C5(); //反向传播
  bool Backward_S4();
  bool Backward_C3();
  bool Backward_S2();
  bool Backward_C1();
  bool Backward_input();

  bool UpdateWeights(); //更新权值、阈值
  void update_weights_bias(const float *delta, float *e_weight, float *weight,
                           int len);

  float test(); //训练完一次计算一次准确率

  //
  bool bmp8(const float *data, int width, int height, const char *name);

#ifdef FORWARD_GPU
  char* load_kernel_source(std::string filename);
#endif

private:
  float *data_input_train;  //原始标准输入数据，训练,范围：[-1, 1]
  float *data_output_train; //原始标准期望结果，训练,取值：-0.8/0.8
  float *data_input_test;   //原始标准输入数据，测试,范围：[-1, 1]
  float *data_output_test; //原始标准期望结果，测试,取值：-0.8/0.8
  float *data_single_image;
  float *data_single_label;

  float weight_C1[len_weight_C1_CNN];
  float bias_C1[len_bias_C1_CNN];
  float weight_S2[len_weight_S2_CNN];
  float bias_S2[len_bias_S2_CNN];
  float weight_C3[len_weight_C3_CNN];
  float bias_C3[len_bias_C3_CNN];
  float weight_S4[len_weight_S4_CNN];
  float bias_S4[len_bias_S4_CNN];
  float weight_C5[len_weight_C5_CNN];
  float bias_C5[len_bias_C5_CNN];
  float weight_output[len_weight_output_CNN];
  float bias_output[len_bias_output_CNN];

  float E_weight_C1[len_weight_C1_CNN]; //累积误差
  float E_bias_C1[len_bias_C1_CNN];
  float E_weight_S2[len_weight_S2_CNN];
  float E_bias_S2[len_bias_S2_CNN];
  float E_weight_C3[len_weight_C3_CNN];
  float E_bias_C3[len_bias_C3_CNN];
  float E_weight_S4[len_weight_S4_CNN];
  float E_bias_S4[len_bias_S4_CNN];
  float *E_weight_C5;
  float *E_bias_C5;
  float *E_weight_output;
  float *E_bias_output;

  float neuron_input[num_neuron_input_CNN]; // data_single_image
  float neuron_C1[num_neuron_C1_CNN];
  float neuron_S2[num_neuron_S2_CNN];
  float neuron_C3[num_neuron_C3_CNN];
  float neuron_S4[num_neuron_S4_CNN];
  float neuron_C5[num_neuron_C5_CNN];
  float neuron_output[num_neuron_output_CNN];

  float delta_neuron_output[num_neuron_output_CNN]; //神经元误差
  float delta_neuron_C5[num_neuron_C5_CNN];
  float delta_neuron_S4[num_neuron_S4_CNN];
  float delta_neuron_C3[num_neuron_C3_CNN];
  float delta_neuron_S2[num_neuron_S2_CNN];
  float delta_neuron_C1[num_neuron_C1_CNN];
  float delta_neuron_input[num_neuron_input_CNN];

  float delta_weight_C1[len_weight_C1_CNN]; //权值、阈值误差
  float delta_bias_C1[len_bias_C1_CNN];
  float delta_weight_S2[len_weight_S2_CNN];
  float delta_bias_S2[len_bias_S2_CNN];
  float delta_weight_C3[len_weight_C3_CNN];
  float delta_bias_C3[len_bias_C3_CNN];
  float delta_weight_S4[len_weight_S4_CNN];
  float delta_bias_S4[len_bias_S4_CNN];
  float delta_weight_C5[len_weight_C5_CNN];
  float delta_bias_C5[len_bias_C5_CNN];
  float delta_weight_output[len_weight_output_CNN];
  float delta_bias_output[len_bias_output_CNN];

#ifdef FORWARD_GPU
  int device_id = 0;
  int platform_id = 0;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context = NULL;
  cl_command_queue cmdQueue;
#ifdef GPU_OPT_LOCAL_MEM
  const char *forward_source_C1_name = "kernel/kernel_forward_C1_local_mem.cl";
  const char *forward_source_S2_name = "kernel/kernel_forward_S2_local_mem.cl";
  const char *forward_source_C3_name = "kernel/kernel_forward_C3_local_mem.cl";
  const char *forward_source_S4_name = "kernel/kernel_forward_S4.cl";
  const char *forward_source_C5_name = "kernel/kernel_forward_C5.cl";
  const char *forward_source_output_name = "kernel/kernel_forward_output.cl";
  const char *backward_source_C1_input_name = "kernel/kernel_backward_C1_input.cl";
  const char *backward_source_C1_weight_name = "kernel/kernel_backward_C1_weight.cl";
  const char *backward_source_S2_input_name = "kernel/kernel_backward_S2_input.cl";
  const char *backward_source_S2_weight_name = "kernel/kernel_backward_S2_weight.cl";
  const char *backward_source_C3_input_name = "kernel/kernel_backward_C3_input.cl";
  const char *backward_source_C3_weight_name = "kernel/kernel_backward_C3_weight.cl";
  // const char *backward_source_C1_input_name = "kernel/kernel_backward_C1_input_tiling.cl";
  // const char *backward_source_C1_weight_name = "kernel/kernel_backward_C1_weight_tiling.cl";
  // const char *backward_source_S2_input_name = "kernel/kernel_backward_S2_input_tiling.cl";
  // const char *backward_source_S2_weight_name = "kernel/kernel_backward_S2_weight_tiling.cl";
  // const char *backward_source_C3_input_name = "kernel/kernel_backward_C3_input_tiling.cl";
  // const char *backward_source_C3_weight_name = "kernel/kernel_backward_C3_weight_tiling.cl";
  const char *backward_source_S4_input_name = "kernel/kernel_backward_S4_input.cl";
  const char *backward_source_S4_weight_name = "kernel/kernel_backward_S4_weight.cl";
  const char *backward_source_C5_name = "kernel/kernel_backward_C5.cl";
  const char *backward_source_output_name = "kernel/kernel_backward_output.cl";
  const char *backward_source_input_weight_name = "kernel/kernel_backward_input_weight.cl";
  const char *update_source_name = "kernel/kernel_update.cl";
#else
  const char *forward_source_C1_name = "kernel/kernel_forward_C1.cl";
  const char *forward_source_S2_name = "kernel/kernel_forward_S2.cl";
  const char *forward_source_C3_name = "kernel/kernel_forward_C3.cl";
  const char *forward_source_S4_name = "kernel/kernel_forward_S4.cl";
  const char *forward_source_C5_name = "kernel/kernel_forward_C5.cl";
  const char *forward_source_output_name = "kernel/kernel_forward_output.cl";
  const char *backward_source_C1_input_name = "kernel/kernel_backward_C1_input.cl";
  const char *backward_source_C1_weight_name = "kernel/kernel_backward_C1_weight.cl";
  const char *backward_source_S2_input_name = "kernel/kernel_backward_S2_input.cl";
  const char *backward_source_S2_weight_name = "kernel/kernel_backward_S2_weight.cl";
  const char *backward_source_C3_input_name = "kernel/kernel_backward_C3_input.cl";
  const char *backward_source_C3_weight_name = "kernel/kernel_backward_C3_weight.cl";
  const char *backward_source_S4_input_name = "kernel/kernel_backward_S4_input.cl";
  const char *backward_source_S4_weight_name = "kernel/kernel_backward_S4_weight.cl";
  const char *backward_source_C5_name = "kernel/kernel_backward_C5.cl";
  const char *backward_source_output_name = "kernel/kernel_backward_output.cl";
  const char *backward_source_input_weight_name = "kernel/kernel_backward_input_weight.cl";
  const char *update_source_name = "kernel/kernel_update.cl";
#endif
  char *forward_source[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
  char *backward_source[11] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  char *update_source[1] = {NULL};

  cl_program forward_program_C1;
  cl_program forward_program_S2;
  cl_program forward_program_C3;
  cl_program forward_program_S4;
  cl_program forward_program_C5;
  cl_program forward_program_output;
  cl_program backward_program_input_weight;
  cl_program backward_program_C1_weight;
  cl_program backward_program_S2_weight;
  cl_program backward_program_C3_weight;
  cl_program backward_program_S4_weight;
  cl_program backward_program_C1_input;
  cl_program backward_program_S2_input;
  cl_program backward_program_C3_input;
  cl_program backward_program_S4_input;
  cl_program backward_program_C5;
  cl_program backward_program_output;
  cl_program update_program;

  cl_kernel forward_C1_kernel;
  cl_kernel forward_S2_kernel;
  cl_kernel forward_C3_kernel;
  cl_kernel forward_S4_kernel;
  cl_kernel forward_C5_kernel;
  cl_kernel forward_output_kernel;
  cl_kernel backward_input_weight_kernel;
  cl_kernel backward_C1_weight_kernel;
  cl_kernel backward_C1_input_kernel;
  cl_kernel backward_S2_weight_kernel;
  cl_kernel backward_S2_input_kernel;
  cl_kernel backward_C3_weight_kernel;
  cl_kernel backward_C3_input_kernel;
  cl_kernel backward_S4_weight_kernel;
  cl_kernel backward_S4_input_kernel;
  cl_kernel backward_C5_kernel;
  cl_kernel backward_output_kernel;
  cl_kernel update_kernel;

#ifdef GPU_OPT_LOCAL_MEM
  int forward_kernel_dims[6] = {3, 3, 3, 3, 3, 1};
  // {input_weight, C1_input, C1_weight, S2_input, S2_weight,
  //  C3_input, C3_weight, S4_input, S4_weight, C5, output}
  int backward_kernel_dims[11] = {2, 3, 1, 3, 2,
                                 3, 1, 3, 2, 1, 1};
  int update_kernel_dim = 1;

  size_t forward_C1_kernel_global[3] = {28, 28, 6};
  size_t forward_S2_kernel_global[3] = {14, 14, 6};
  size_t forward_C3_kernel_global[3] = {10, 10, 16};
  size_t forward_S4_kernel_global[3] = {5, 5, 16};
  size_t forward_C5_kernel_global[3] = {1, 1, 120};
  size_t forward_output_kernel_global[1] = {10};

  size_t forward_C1_kernel_local[3] = {7, 7, 1};
  size_t forward_S2_kernel_local[3] = {2, 2, 1};
  size_t forward_C3_kernel_local[3] = {2, 2, 1};
  // size_t forward_S4_kernel_local[3] = {1, 1, 1};
  // size_t forward_C5_kernel_local[3] = {1, 1, 2};
  size_t* forward_S4_kernel_local = NULL;
  size_t* forward_C5_kernel_local = NULL;
  size_t* forward_output_kernel_local = NULL;

  size_t backward_input_weight_kernel_global[2] = {1, 6};
  size_t backward_C1_input_kernel_global[3] = {28, 28, 6};
  size_t backward_C1_weight_kernel_global[3] = {6};
  size_t backward_S2_input_kernel_global[3] = {14, 14, 6};
  size_t backward_S2_weight_kernel_global[2] = {6, 16};
  size_t backward_C3_input_kernel_global[3] = {10, 10, 16};
  size_t backward_C3_weight_kernel_global[1] = {16};
  size_t backward_S4_input_kernel_global[3] = {5, 5, 16};
  size_t backward_S4_weight_kernel_global[2] = {16, 120};
  size_t backward_C5_kernel_global[1] = {120};
  size_t backward_output_kernel_global[1] = {10};

  size_t *backward_input_weight_kernel_local = NULL;
  size_t *backward_C1_input_kernel_local = NULL;
  size_t *backward_C1_weight_kernel_local = NULL;
  size_t *backward_S2_input_kernel_local = NULL;
  size_t *backward_S2_weight_kernel_local = NULL;
  size_t *backward_C3_input_kernel_local = NULL;
  size_t *backward_C3_weight_kernel_local = NULL;
  size_t *backward_S4_input_kernel_local = NULL;
  size_t *backward_S4_weight_kernel_local = NULL;
  size_t *backward_C5_kernel_local = NULL;
  size_t *backward_output_kernel_local = NULL;

  // size_t *backward_input_weight_kernel_local = NULL;
  // size_t backward_C1_input_kernel_local[3] = {2, 2, 2};
  // size_t *backward_C1_weight_kernel_local = NULL;
  // size_t backward_S2_input_kernel_local[3] = {2, 2, 2};
  // size_t backward_S2_weight_kernel_local[2] = {2, 2};
  // size_t backward_C3_input_kernel_local[3] = {2, 2, 2};
  // size_t *backward_C3_weight_kernel_local = NULL;
  // size_t *backward_S4_input_kernel_local = NULL;
  // size_t *backward_S4_weight_kernel_local = NULL;
  // size_t *backward_C5_kernel_local = NULL;
  // size_t *backward_output_kernel_local = NULL;
#else
  int forward_kernel_dims[6] = {3, 3, 3, 3, 3, 1};
  // {input_weight, C1_input, C1_weight, S2_input, S2_weight,
  //  C3_input, C3_weight, S4_input, S4_weight, C5, output}
  int backward_kernel_dims[12] = {2, 3, 1, 3, 2,
                                 3, 1, 3, 2, 1, 1};
  int update_kernel_dim = 1;

  size_t forward_C1_kernel_global[3] = {6, 28, 28};
  size_t forward_S2_kernel_global[3] = {6, 14, 14};
  size_t forward_C3_kernel_global[3] = {16, 10, 10};
  size_t forward_S4_kernel_global[3] = {16, 5, 5};
  size_t forward_C5_kernel_global[3] = {120, 1, 1};
  size_t forward_output_kernel_global[1] = {10};

  size_t* forward_C1_kernel_local = NULL;
  size_t* forward_S2_kernel_local = NULL;
  size_t* forward_C3_kernel_local = NULL;
  size_t* forward_S4_kernel_local = NULL;
  size_t* forward_C5_kernel_local = NULL;
  size_t* forward_output_kernel_local = NULL;

  size_t backward_input_weight_kernel_global[2] = {1, 6};
  size_t backward_C1_input_kernel_global[3] = {28, 28, 6};
  size_t backward_C1_weight_kernel_global[3] = {6};
  size_t backward_S2_input_kernel_global[3] = {14, 14, 6};
  size_t backward_S2_weight_kernel_global[2] = {6, 16};
  size_t backward_C3_input_kernel_global[3] = {10, 10, 16};
  size_t backward_C3_weight_kernel_global[1] = {16};
  size_t backward_S4_input_kernel_global[3] = {5, 5, 16};
  size_t backward_S4_weight_kernel_global[2] = {16, 120};
  size_t backward_C5_kernel_global[1] = {120};
  size_t backward_output_kernel_global[1] = {10};

  size_t *backward_input_weight_kernel_local = NULL;
  size_t *backward_C1_input_kernel_local = NULL;
  size_t *backward_C1_weight_kernel_local = NULL;
  size_t *backward_S2_input_kernel_local = NULL;
  size_t *backward_S2_weight_kernel_local = NULL;
  size_t *backward_C3_input_kernel_local = NULL;
  size_t *backward_C3_weight_kernel_local = NULL;
  size_t *backward_S4_input_kernel_local = NULL;
  size_t *backward_S4_weight_kernel_local = NULL;
  size_t *backward_C5_kernel_local = NULL;
  size_t *backward_output_kernel_local = NULL;
#endif

  cl_mem device_data_input_train;  //原始标准输入数据，训练,范围：[-1, 1]
  cl_mem device_data_output_train; //原始标准期望结果，训练,取值：-0.8/0.8
  cl_mem device_data_input_test;   //原始标准输入数据，测试,范围：[-1, 1]
  cl_mem device_data_output_test; //原始标准期望结果，测试,取值：-0.8/0.8
  cl_mem device_data_pointer;
  cl_mem device_label_pointer;
  int image_offset;
  int label_offset;

  #define O true
  #define X false
  bool host_tbl[6*16] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
  };
  #undef O
  #undef X
  cl_mem device_tbl;

  cl_mem device_weight_C1;
  cl_mem device_bias_C1;
  cl_mem device_weight_S2;
  cl_mem device_bias_S2;
  cl_mem device_weight_C3;
  cl_mem device_bias_C3;
  cl_mem device_weight_S4;
  cl_mem device_bias_S4;
  cl_mem device_weight_C5;
  cl_mem device_bias_C5;
  cl_mem device_weight_output;
  cl_mem device_bias_output;

  cl_mem device_E_weight_C1; //累积误差
  cl_mem device_E_bias_C1;
  cl_mem device_E_weight_S2;
  cl_mem device_E_bias_S2;
  cl_mem device_E_weight_C3;
  cl_mem device_E_bias_C3;
  cl_mem device_E_weight_S4;
  cl_mem device_E_bias_S4;
  cl_mem device_E_weight_C5;
  cl_mem device_E_bias_C5;
  cl_mem device_E_weight_output;
  cl_mem device_E_bias_output;

//   cl_mem device_neuron_input; // data_single_image
  cl_mem device_neuron_C1;
  cl_mem device_neuron_S2;
  cl_mem device_neuron_C3;
  cl_mem device_neuron_S4;
  cl_mem device_neuron_C5;
  cl_mem device_neuron_output;

  cl_mem device_delta_neuron_output; //神经元误差
  cl_mem device_delta_neuron_C5;
  cl_mem device_delta_neuron_S4;
  cl_mem device_delta_neuron_C3;
  cl_mem device_delta_neuron_S2;
  cl_mem device_delta_neuron_C1;
  cl_mem device_delta_neuron_input;

  cl_mem device_delta_weight_C1; //权值、阈值误差
  cl_mem device_delta_bias_C1;
  cl_mem device_delta_weight_S2;
  cl_mem device_delta_bias_S2;
  cl_mem device_delta_weight_C3;
  cl_mem device_delta_bias_C3;
  cl_mem device_delta_weight_S4;
  cl_mem device_delta_bias_S4;
  cl_mem device_delta_weight_C5;
  cl_mem device_delta_bias_C5;
  cl_mem device_delta_weight_output;
  cl_mem device_delta_bias_output;
#endif
};

#endif /* CNN_H_ */
