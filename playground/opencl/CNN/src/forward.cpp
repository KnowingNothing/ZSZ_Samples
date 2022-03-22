/*
 * forward.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"
#ifndef FORWARD_GPU
#include <immintrin.h>
#include <omp.h>

using namespace std;

// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
static const bool tbl[6][16] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O, O, X, X, X, O, O, O,
    X, X, O, O, O, O, X, O, O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O, X, X, O, O, O, X, X, O,
    O, O, O, X, O, O, X, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O};
#undef O
#undef X

// #define UNROLL
// #define UNROLL_VECTORIZE
#define UNROLL_VECTORIZE_PARALLEL

#if defined(UNROLL_VECTORIZE_PARALLEL)
bool CNN::Forward_C1(bool train) {
#pragma omp parallel for
  for (int channel = 0; channel < num_map_C1_CNN; channel++) {
    for (int y = 0; y < height_image_C1_CNN; y++) {
      for (int x = 0; x < width_image_C1_CNN; x++) {
        int index = (channel * height_image_C1_CNN * width_image_C1_CNN) +
                    y * width_image_C1_CNN + x; //当前神经元
        neuron_C1[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_input_CNN; inc++) {
          int addr1 = get_index(0, 0, num_map_input_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C1_CNN * num_map_input_CNN);
          int addr2 = get_index(0, 0, inc, width_image_input_CNN,
                                height_image_input_CNN, num_map_input_CNN);
          const float *pw = &weight_C1[0] + addr1;     //卷积核
          const float *pi = data_single_image + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_input_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              sum += *ppw++ * ppi[wy * width_image_input_CNN + wx];
            }
          }
          neuron_C1[index] += sum;
        }
        neuron_C1[index] += bias_C1[channel]; //加偏置
        neuron_C1[index] =
            activation_function_tanh(neuron_C1[index]); //激励函数
      }
    }
  }
  return true;
}
#elif defined(UNROLL_VECTORIZE)
bool CNN::Forward_C1() {
  for (int y = 0; y < height_image_C1_CNN; y++) {
    for (int x = 0; x < width_image_C1_CNN; x++) {
      __m256 sum_vec = _mm256_setzero_ps();
      float *ptr_sum = (float *)&sum_vec;
      //卷积运算
      for (int inc = 0; inc < num_map_input_CNN; inc++) {
        for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
          for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
            float tmp_weight[8] = {0.0};
            tmp_weight[wx] = weight_C1
                [((0 * num_map_input_CNN + inc) * height_kernel_conv_CNN + wy) *
                     width_kernel_conv_CNN +
                 wx];
            tmp_weight[wx] = weight_C1
                [((1 * num_map_input_CNN + inc) * height_kernel_conv_CNN + wy) *
                     width_kernel_conv_CNN +
                 wx];
            tmp_weight[wx] = weight_C1
                [((2 * num_map_input_CNN + inc) * height_kernel_conv_CNN + wy) *
                     width_kernel_conv_CNN +
                 wx];
            tmp_weight[wx] = weight_C1
                [((3 * num_map_input_CNN + inc) * height_kernel_conv_CNN + wy) *
                     width_kernel_conv_CNN +
                 wx];
            tmp_weight[wx] = weight_C1
                [((4 * num_map_input_CNN + inc) * height_kernel_conv_CNN + wy) *
                     width_kernel_conv_CNN +
                 wx];
            tmp_weight[wx] = weight_C1
                [((5 * num_map_input_CNN + inc) * height_kernel_conv_CNN + wy) *
                     width_kernel_conv_CNN +
                 wx];
            __m256 weight_vec = _mm256_loadu_ps(tmp_weight);
            __m256 image_vec =
                _mm256_broadcast_ss(data_single_image +
                                    (inc * height_image_input_CNN + (y + wy)) *
                                        width_image_input_CNN +
                                    (x + wx));
            __m256 tmp_mul = _mm256_mul_ps(image_vec, weight_vec);
            sum_vec = _mm256_add_ps(sum_vec, tmp_mul);
          }
        }
      }
      int index = (0 * height_image_C1_CNN * width_image_C1_CNN) +
                  y * width_image_C1_CNN + x; //当前神经元
      neuron_C1[index] = sum_vec[0] + bias_C1[0];
      neuron_C1[index] = activation_function_tanh(neuron_C1[index]); //激励函数
      index += height_image_C1_CNN * width_image_C1_CNN;
      neuron_C1[index] = sum_vec[1] + bias_C1[1]; //激励函数
      neuron_C1[index] = activation_function_tanh(neuron_C1[index]); //激励函数
      index += height_image_C1_CNN * width_image_C1_CNN;
      neuron_C1[index] = sum_vec[2] + bias_C1[2]; //激励函数
      neuron_C1[index] = activation_function_tanh(neuron_C1[index]); //激励函数
      index += height_image_C1_CNN * width_image_C1_CNN;
      neuron_C1[index] = sum_vec[3] + bias_C1[3]; //激励函数
      neuron_C1[index] = activation_function_tanh(neuron_C1[index]); //激励函数
      index += height_image_C1_CNN * width_image_C1_CNN;
      neuron_C1[index] = sum_vec[4] + bias_C1[4]; //激励函数
      neuron_C1[index] = activation_function_tanh(neuron_C1[index]); //激励函数
      index += height_image_C1_CNN * width_image_C1_CNN;
      neuron_C1[index] = sum_vec[5] + bias_C1[5]; //激励函数
      neuron_C1[index] = activation_function_tanh(neuron_C1[index]); //激励函数
    }
  }

  return true;
}
#elif defined(UNROLL)
bool CNN::Forward_C1() {
  for (int channel = 0; channel < num_map_C1_CNN; channel++) {
    for (int y = 0; y < height_image_C1_CNN; y++) {
      for (int x = 0; x < width_image_C1_CNN; x++) {
        int index = (channel * height_image_C1_CNN * width_image_C1_CNN) +
                    y * width_image_C1_CNN + x; //当前神经元
        neuron_C1[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_input_CNN; inc++) {
          int addr1 = get_index(0, 0, num_map_input_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C1_CNN * num_map_input_CNN);
          int addr2 = get_index(0, 0, inc, width_image_input_CNN,
                                height_image_input_CNN, num_map_input_CNN);
          const float *pw = &weight_C1[0] + addr1;     //卷积核
          const float *pi = data_single_image + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_input_CNN + x;

          sum += *ppw++ * ppi[0 * width_image_input_CNN + 0];
          sum += *ppw++ * ppi[0 * width_image_input_CNN + 1];
          sum += *ppw++ * ppi[0 * width_image_input_CNN + 2];
          sum += *ppw++ * ppi[0 * width_image_input_CNN + 3];
          sum += *ppw++ * ppi[0 * width_image_input_CNN + 4];

          sum += *ppw++ * ppi[1 * width_image_input_CNN + 0];
          sum += *ppw++ * ppi[1 * width_image_input_CNN + 1];
          sum += *ppw++ * ppi[1 * width_image_input_CNN + 2];
          sum += *ppw++ * ppi[1 * width_image_input_CNN + 3];
          sum += *ppw++ * ppi[1 * width_image_input_CNN + 4];

          sum += *ppw++ * ppi[2 * width_image_input_CNN + 0];
          sum += *ppw++ * ppi[2 * width_image_input_CNN + 1];
          sum += *ppw++ * ppi[2 * width_image_input_CNN + 2];
          sum += *ppw++ * ppi[2 * width_image_input_CNN + 3];
          sum += *ppw++ * ppi[2 * width_image_input_CNN + 4];

          sum += *ppw++ * ppi[3 * width_image_input_CNN + 0];
          sum += *ppw++ * ppi[3 * width_image_input_CNN + 1];
          sum += *ppw++ * ppi[3 * width_image_input_CNN + 2];
          sum += *ppw++ * ppi[3 * width_image_input_CNN + 3];
          sum += *ppw++ * ppi[3 * width_image_input_CNN + 4];

          sum += *ppw++ * ppi[4 * width_image_input_CNN + 0];
          sum += *ppw++ * ppi[4 * width_image_input_CNN + 1];
          sum += *ppw++ * ppi[4 * width_image_input_CNN + 2];
          sum += *ppw++ * ppi[4 * width_image_input_CNN + 3];
          sum += *ppw++ * ppi[4 * width_image_input_CNN + 4];
          neuron_C1[index] += sum;
        }
        neuron_C1[index] += bias_C1[channel]; //加偏置
        neuron_C1[index] =
            activation_function_tanh(neuron_C1[index]); //激励函数
      }
    }
  }
  return true;
}
#else
bool CNN::Forward_C1() {
  for (int channel = 0; channel < num_map_C1_CNN; channel++) {
    for (int y = 0; y < height_image_C1_CNN; y++) {
      for (int x = 0; x < width_image_C1_CNN; x++) {
        int index = (channel * height_image_C1_CNN * width_image_C1_CNN) +
                    y * width_image_C1_CNN + x; //当前神经元
        neuron_C1[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_input_CNN; inc++) {
          int addr1 = get_index(0, 0, num_map_input_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C1_CNN * num_map_input_CNN);
          int addr2 = get_index(0, 0, inc, width_image_input_CNN,
                                height_image_input_CNN, num_map_input_CNN);
          const float *pw = &weight_C1[0] + addr1;     //卷积核
          const float *pi = data_single_image + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_input_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              sum += *ppw++ * ppi[wy * width_image_input_CNN + wx];
            }
          }
          neuron_C1[index] += sum;
        }
        neuron_C1[index] += bias_C1[channel]; //加偏置
        neuron_C1[index] =
            activation_function_tanh(neuron_C1[index]); //激励函数
      }
    }
  }
  return true;
}
#endif

bool CNN::Forward_S2() {
  float scale_factor =
      1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

  for (int i = 0; i < num_map_S2_CNN; i++) {
    int block = width_image_C1_CNN * height_image_C1_CNN * i;
    for (int y = 0; y < height_image_S2_CNN; y++) {
      for (int x = 0; x < width_image_S2_CNN; x++) {
        int rows = y * width_kernel_pooling_CNN;
        int cols = x * height_kernel_pooling_CNN;
        int index = (i * height_image_S2_CNN * width_image_S2_CNN) +
                    y * width_image_S2_CNN + x;

        neuron_S2[index] = 0.0;
        for (int m = 0; m < width_kernel_pooling_CNN; m++) {
          for (int n = 0; n < height_kernel_pooling_CNN; n++) {
            neuron_S2[index] +=
                weight_S2[i] *
                neuron_C1[(rows + m) * width_image_C1_CNN + cols + n + block];
          }
        }
        //
        neuron_S2[index] *= scale_factor;
        neuron_S2[index] += bias_S2[i];
        neuron_S2[index] = activation_function_tanh(neuron_S2[index]);
      }
    }
  }
  return true;
}

#if defined(UNROLL_VECTORIZE_PARALLEL)
bool CNN::Forward_C3() {
#pragma omp parallel for
  for (int channel = 0; channel < num_map_C3_CNN; channel++) {
    for (int y = 0; y < height_image_C3_CNN; y++) {
      for (int x = 0; x < width_image_C3_CNN; x++) {
        int index = (channel * height_image_C3_CNN * width_image_C3_CNN) +
                    y * width_image_C3_CNN + x; //当前神经元
        neuron_C3[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S2_CNN; inc++) {
          if (!tbl[inc][channel])
            continue;
          int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C3_CNN * num_map_S2_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S2_CNN,
                                height_image_S2_CNN, num_map_S2_CNN); //输入图像
          const float *pw = &weight_C3[0] + addr1; //卷积核
          const float *pi = &neuron_S2[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S2_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              sum += *ppw++ * ppi[wy * width_image_S2_CNN + wx];
            }
          }
          neuron_C3[index] += sum;
        }
        neuron_C3[index] += bias_C3[channel]; //加偏置
        neuron_C3[index] =
            activation_function_tanh(neuron_C3[index]); //激励函数
      }
    }
  }
  return true;
}
#elif defined(UNROLL_VECTORIZE)
bool CNN::Forward_C3() {
  for (int channel = 0; channel < num_map_C3_CNN; channel++) {
    for (int y = 0; y < height_image_C3_CNN; y++) {
      for (int x = 0; x < width_image_C3_CNN; x++) {
        int index = (channel * height_image_C3_CNN * width_image_C3_CNN) +
                    y * width_image_C3_CNN + x; //当前神经元
        neuron_C3[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S2_CNN; inc++) {
          if (!tbl[inc][channel])
            continue;
          int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C3_CNN * num_map_S2_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S2_CNN,
                                height_image_S2_CNN, num_map_S2_CNN); //输入图像
          const float *pw = &weight_C3[0] + addr1; //卷积核
          const float *pi = &neuron_S2[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S2_CNN + x;
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 4];
          neuron_C3[index] += sum;
        }
        neuron_C3[index] += bias_C3[channel]; //加偏置
        neuron_C3[index] =
            activation_function_tanh(neuron_C3[index]); //激励函数
      }
    }
  }
  return true;
}
#elif defined(UNROLL)
bool CNN::Forward_C3() {
  for (int channel = 0; channel < num_map_C3_CNN; channel++) {
    for (int y = 0; y < height_image_C3_CNN; y++) {
      for (int x = 0; x < width_image_C3_CNN; x++) {
        int index = (channel * height_image_C3_CNN * width_image_C3_CNN) +
                    y * width_image_C3_CNN + x; //当前神经元
        neuron_C3[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S2_CNN; inc++) {
          if (!tbl[inc][channel])
            continue;
          int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C3_CNN * num_map_S2_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S2_CNN,
                                height_image_S2_CNN, num_map_S2_CNN); //输入图像
          const float *pw = &weight_C3[0] + addr1; //卷积核
          const float *pi = &neuron_S2[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S2_CNN + x;
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[0 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[1 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[2 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[3 * width_image_S2_CNN + 4];

          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 0];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 1];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 2];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 3];
          sum += *ppw++ * ppi[4 * width_image_S2_CNN + 4];
          neuron_C3[index] += sum;
        }
        neuron_C3[index] += bias_C3[channel]; //加偏置
        neuron_C3[index] =
            activation_function_tanh(neuron_C3[index]); //激励函数
      }
    }
  }
  return true;
}
#else
bool CNN::Forward_C3() {
  for (int channel = 0; channel < num_map_C3_CNN; channel++) {
    for (int y = 0; y < height_image_C3_CNN; y++) {
      for (int x = 0; x < width_image_C3_CNN; x++) {
        int index = (channel * height_image_C3_CNN * width_image_C3_CNN) +
                    y * width_image_C3_CNN + x; //当前神经元
        neuron_C3[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S2_CNN; inc++) {
          if (!tbl[inc][channel])
            continue;
          int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C3_CNN * num_map_S2_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S2_CNN,
                                height_image_S2_CNN, num_map_S2_CNN); //输入图像
          const float *pw = &weight_C3[0] + addr1; //卷积核
          const float *pi = &neuron_S2[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S2_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              sum += *ppw++ * ppi[wy * width_image_S2_CNN + wx];
            }
          }
          neuron_C3[index] += sum;
        }
        neuron_C3[index] += bias_C3[channel]; //加偏置
        neuron_C3[index] =
            activation_function_tanh(neuron_C3[index]); //激励函数
      }
    }
  }
  return true;
}
#endif

bool CNN::Forward_S4() {
  float scale_factor =
      1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
  for (int i = 0; i < num_map_S4_CNN; i++) {
    int block = width_image_C3_CNN * height_image_C3_CNN * i; // C3
    for (int y = 0; y < height_image_S4_CNN; y++) {
      for (int x = 0; x < width_image_S4_CNN; x++) {
        int rows = y * width_kernel_pooling_CNN;
        int cols = x * height_kernel_pooling_CNN;
        int index = (i * height_image_S4_CNN * width_image_S4_CNN) +
                    y * width_image_S4_CNN + x; // S4 当前神经元

        neuron_S4[index] = 0.0;
        for (int m = 0; m < width_kernel_pooling_CNN; m++) {
          for (int n = 0; n < height_kernel_pooling_CNN; n++) {
            neuron_S4[index] +=
                weight_S4[i] *
                neuron_C3[(rows + m) * width_image_C3_CNN + cols + n + block];
          }
        }
        //
        neuron_S4[index] *= scale_factor;
        neuron_S4[index] += bias_S4[i];
        neuron_S4[index] = activation_function_tanh(neuron_S4[index]);
      }
    }
  }
  return true;
}

#if defined(UNROLL_VECTORIZE_PARALLEL)
bool CNN::Forward_C5() {
#pragma omp parallel for
  for (int channel = 0; channel < num_map_C5_CNN; channel++) {
    for (int y = 0; y < height_image_C5_CNN; y++) {
      for (int x = 0; x < width_image_C5_CNN; x++) {
        int index = (channel * height_image_C5_CNN * width_image_C5_CNN) +
                    y * width_image_C5_CNN + x; //当前神经元
        neuron_C5[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S4_CNN; inc++) {
          int addr1 = get_index(0, 0, num_map_S4_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C5_CNN * num_map_S4_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S4_CNN,
                                height_image_S4_CNN, num_map_S4_CNN);
          const float *pw = &weight_C5[0] + addr1; //卷积核
          const float *pi = &neuron_S4[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S4_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
            }
          }
          neuron_C5[index] += sum;
        }
        neuron_C5[index] += bias_C5[channel]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
      }
    }
  }
  return true;
}
#elif defined(UNROLL_VECTORIZE)
bool CNN::Forward_C5() {
  for (int c_outer = 0; c_outer < 15; ++c_outer) {
    for (int y = 0; y < height_image_C5_CNN; y++) {
      for (int x = 0; x < width_image_C5_CNN; x++) {
        __m256 sum_vec;
        sum_vec = _mm256_setzero_ps();
        //卷积运算
        for (int inc = 0; inc < num_map_S4_CNN; inc++) {
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              __m256 weight_vec;
              float tmp_weight[8] = {0.0};
              __m256 image_vec = _mm256_broadcast_ss(
                  neuron_S4 +
                  (inc * height_image_S4_CNN + (y + wy)) * width_image_S4_CNN +
                  (x + wx));
              for (int c_inner = 0; c_inner < 8; c_inner++) {
                int channel = c_outer * 8 + c_inner;
                tmp_weight[c_inner] =
                    weight_C5[((channel * num_map_input_CNN + inc) *
                                   height_kernel_conv_CNN +
                               wy) *
                                  width_kernel_conv_CNN +
                              wx];
              }
              weight_vec = _mm256_loadu_ps(tmp_weight);
              __m256 tmp_mul = _mm256_mul_ps(image_vec, weight_vec);
              sum_vec = _mm256_add_ps(tmp_mul, sum_vec);
            }
          }
        }
        int index = (c_outer * 8 * height_image_C5_CNN * width_image_C5_CNN) +
                    y * width_image_C5_CNN + x;
        int channel = c_outer * 8;
        float *ptr_sum = (float *)&sum_vec;
        neuron_C5[index] = ptr_sum[0] + bias_C5[channel + 0]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[1] + bias_C5[channel + 1]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[2] + bias_C5[channel + 2]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[3] + bias_C5[channel + 3]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[4] + bias_C5[channel + 4]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[5] + bias_C5[channel + 5]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[6] + bias_C5[channel + 6]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
        neuron_C5[index] = ptr_sum[7] + bias_C5[channel + 7]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
        index += height_image_C5_CNN * width_image_C5_CNN;
      }
    }
  }
  return true;
}
#elif defined(UNROLL)
bool CNN::Forward_C5() {
  for (int channel = 0; channel < num_map_C5_CNN; channel++) {
    for (int y = 0; y < height_image_C5_CNN; y++) {
      for (int x = 0; x < width_image_C5_CNN; x++) {
        int index = (channel * height_image_C5_CNN * width_image_C5_CNN) +
                    y * width_image_C5_CNN + x; //当前神经元
        neuron_C5[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S4_CNN; inc++) {
          int addr1 = get_index(0, 0, num_map_S4_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C5_CNN * num_map_S4_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S4_CNN,
                                height_image_S4_CNN, num_map_S4_CNN);
          const float *pw = &weight_C5[0] + addr1; //卷积核
          const float *pi = &neuron_S4[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S4_CNN + x;
          sum += *ppw++ * ppi[0 * width_image_S4_CNN + 0];
          sum += *ppw++ * ppi[0 * width_image_S4_CNN + 1];
          sum += *ppw++ * ppi[0 * width_image_S4_CNN + 2];
          sum += *ppw++ * ppi[0 * width_image_S4_CNN + 3];
          sum += *ppw++ * ppi[0 * width_image_S4_CNN + 4];

          sum += *ppw++ * ppi[1 * width_image_S4_CNN + 0];
          sum += *ppw++ * ppi[1 * width_image_S4_CNN + 1];
          sum += *ppw++ * ppi[1 * width_image_S4_CNN + 2];
          sum += *ppw++ * ppi[1 * width_image_S4_CNN + 3];
          sum += *ppw++ * ppi[1 * width_image_S4_CNN + 4];

          sum += *ppw++ * ppi[2 * width_image_S4_CNN + 0];
          sum += *ppw++ * ppi[2 * width_image_S4_CNN + 1];
          sum += *ppw++ * ppi[2 * width_image_S4_CNN + 2];
          sum += *ppw++ * ppi[2 * width_image_S4_CNN + 3];
          sum += *ppw++ * ppi[2 * width_image_S4_CNN + 4];

          sum += *ppw++ * ppi[3 * width_image_S4_CNN + 0];
          sum += *ppw++ * ppi[3 * width_image_S4_CNN + 1];
          sum += *ppw++ * ppi[3 * width_image_S4_CNN + 2];
          sum += *ppw++ * ppi[3 * width_image_S4_CNN + 3];
          sum += *ppw++ * ppi[3 * width_image_S4_CNN + 4];

          sum += *ppw++ * ppi[4 * width_image_S4_CNN + 0];
          sum += *ppw++ * ppi[4 * width_image_S4_CNN + 1];
          sum += *ppw++ * ppi[4 * width_image_S4_CNN + 2];
          sum += *ppw++ * ppi[4 * width_image_S4_CNN + 3];
          sum += *ppw++ * ppi[4 * width_image_S4_CNN + 4];
          neuron_C5[index] += sum;
        }
        neuron_C5[index] += bias_C5[channel]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
      }
    }
  }
  return true;
}
#else
bool CNN::Forward_C5() {
#if 1
  for (int channel = 0; channel < num_map_C5_CNN; channel++) {
    for (int y = 0; y < height_image_C5_CNN; y++) {
      for (int x = 0; x < width_image_C5_CNN; x++) {
        int index = (channel * height_image_C5_CNN * width_image_C5_CNN) +
                    y * width_image_C5_CNN + x; //当前神经元
        neuron_C5[index] = 0.0;
        //卷积运算
        for (int inc = 0; inc < num_map_S4_CNN; inc++) {
          int addr1 = get_index(0, 0, num_map_S4_CNN * channel + inc,
                                width_kernel_conv_CNN, height_kernel_conv_CNN,
                                num_map_C5_CNN * num_map_S4_CNN);
          int addr2 = get_index(0, 0, inc, width_image_S4_CNN,
                                height_image_S4_CNN, num_map_S4_CNN);
          const float *pw = &weight_C5[0] + addr1; //卷积核
          const float *pi = &neuron_S4[0] + addr2; //输入图像
          float sum = 0.0;
          const float *ppw = pw;
          const float *ppi = pi + y * width_image_S4_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
            }
          }
          neuron_C5[index] += sum;
        }
        neuron_C5[index] += bias_C5[channel]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
      }
    }
  }
#else
  for (int channel = 0; channel < num_map_C5_CNN; channel++) {
    for (int y = 0; y < height_image_C5_CNN; y++) {
      for (int x = 0; x < width_image_C5_CNN; x++) {
        int index = (channel * height_image_C5_CNN * width_image_C5_CNN) +
                    y * width_image_C5_CNN + x; // C5 当前神经元
        for (int inc = 0; inc < num_map_S4_CNN; inc++) {
          int addr1 = width_kernel_conv_CNN * height_kernel_conv_CNN *
                      (num_map_S4_CNN * channel + inc); //找到对应的卷积核
          int addr2 =
              height_image_S4_CNN * width_image_S4_CNN * inc; //找到对应的S4输入
          addr2 += y * width_image_S4_CNN + x;
          // const float* pw = &weight_C5[0] + addr1;       //卷积核
          // const float* pi = &neuron_S4[0] + addr2;       //输入图像
          float sum = 0.0;
          // const float* ppw = pw;
          // const float* ppi = pi + y * width_image_S4_CNN + x;
          for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
              int addr3 = wy * width_kernel_conv_CNN + wx; //卷积核索引
              int addr4 = wy * width_image_S4_CNN + wx; // S4中的像素索引
              sum += weight_C5[addr1 + addr3] * neuron_S4[addr2 + addr4];
              // sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
            }
          }
          neuron_C5[index] += sum;
        }
        neuron_C5[index] += bias_C5[channel]; //加偏置
        neuron_C5[index] =
            activation_function_tanh(neuron_C5[index]); //激励函数
      }
    }
  }
#endif
  return true;
}
#endif

bool CNN::Forward_output() {
  for (int i = 0; i < num_neuron_output_CNN; i++) {
    neuron_output[i] = 0.0;
    for (int c = 0; c < num_neuron_C5_CNN; c++) {
      neuron_output[i] +=
          weight_output[c * num_neuron_output_CNN + i] * neuron_C5[c];
    }
    neuron_output[i] += bias_output[i];
    neuron_output[i] = activation_function_tanh(neuron_output[i]);
  }
  return true;
}
#else

using namespace std;

bool CNN::Forward_C1(bool train) {
  cl_int status;
  if (train) {
  status = clSetKernelArg(this->forward_C1_kernel, 0, sizeof(cl_mem),
                          &this->device_data_input_train);
  } else {
status = clSetKernelArg(this->forward_C1_kernel, 0, sizeof(cl_mem),
                          &this->device_data_input_test);
  }
  CHECK_CL(status, "Can't assign args0");
  status |= clSetKernelArg(this->forward_C1_kernel, 1, sizeof(cl_mem),
                           &this->device_weight_C1);
  CHECK_CL(status, "Can't assign args1");
  status |= clSetKernelArg(this->forward_C1_kernel, 2, sizeof(cl_mem),
                           &this->device_bias_C1);
  CHECK_CL(status, "Can't assign args2");
  status |= clSetKernelArg(this->forward_C1_kernel, 3, sizeof(cl_mem),
                           &this->device_neuron_C1);
  CHECK_CL(status, "Can't assign args3");
  status = clSetKernelArg(this->forward_C1_kernel, 4, sizeof(cl_int),
                          &this->image_offset);
  CHECK_CL(status, "Can't assign args4");

  cl_event event;
  //   std::cout << "forward C1 global: "
  //   			<< this->forward_C1_kernel_global[0] << ","
  //             << this->forward_C1_kernel_global[1] << ","
  //             << this->forward_C1_kernel_global[2] << "\n"
  //             << this->forward_kernel_dims[0] << "\n"
  //             << std::flush;
  status = clEnqueueNDRangeKernel(
      this->cmdQueue, this->forward_C1_kernel, this->forward_kernel_dims[0],
      NULL, this->forward_C1_kernel_global, this->forward_C1_kernel_local, 0,
      NULL, &event);
  cl_event waits[] = {event};
  // clWaitForEvents(1, waits);
  clReleaseEvent(event);
  return true;
}

bool CNN::Forward_S2() {
  cl_int status;
  status = clSetKernelArg(this->forward_S2_kernel, 0, sizeof(cl_mem),
                          &this->device_neuron_C1);
  CHECK_CL(status, "Can't assign args0");
  status |= clSetKernelArg(this->forward_S2_kernel, 1, sizeof(cl_mem),
                           &this->device_weight_S2);
  CHECK_CL(status, "Can't assign args1");
  status |= clSetKernelArg(this->forward_S2_kernel, 2, sizeof(cl_mem),
                           &this->device_bias_S2);
  CHECK_CL(status, "Can't assign args2");
  status |= clSetKernelArg(this->forward_S2_kernel, 3, sizeof(cl_mem),
                           &this->device_neuron_S2);
  CHECK_CL(status, "Can't assign args3");

  cl_event event;
  //   std::cout << "forward S2 global: "
  //   			<< this->forward_S2_kernel_global[0] << ","
  //             << this->forward_S2_kernel_global[1] << ","
  //             << this->forward_S2_kernel_global[2] << "\n"
  //             << this->forward_kernel_dims[1] << "\n"
  //             << std::flush;
  status = clEnqueueNDRangeKernel(
      this->cmdQueue, this->forward_S2_kernel, this->forward_kernel_dims[1],
      NULL, this->forward_S2_kernel_global, this->forward_S2_kernel_local, 0,
      NULL, &event);
  cl_event waits[] = {event};
  // clWaitForEvents(1, waits);
  clReleaseEvent(event);
  return true;
}

bool CNN::Forward_C3() {
  cl_int status;
  status = clSetKernelArg(this->forward_C3_kernel, 0, sizeof(cl_mem),
                          &this->device_neuron_S2);
  CHECK_CL(status, "Can't assign args0");
  status |= clSetKernelArg(this->forward_C3_kernel, 1, sizeof(cl_mem),
                           &this->device_weight_C3);
  CHECK_CL(status, "Can't assign args1");
  status |= clSetKernelArg(this->forward_C3_kernel, 2, sizeof(cl_mem),
                           &this->device_bias_C3);
  CHECK_CL(status, "Can't assign args2");
  status |= clSetKernelArg(this->forward_C3_kernel, 3, sizeof(cl_mem),
                           &this->device_neuron_C3);
  CHECK_CL(status, "Can't assign args3");
  status |= clSetKernelArg(this->forward_C3_kernel, 4, sizeof(cl_mem),
                           &this->device_tbl);
  CHECK_CL(status, "Can't assign args4");

  cl_event event;
  //   std::cout << "forward C3 global: "
  //   			<< this->forward_C3_kernel_global[0] << ","
  //             << this->forward_C3_kernel_global[1] << ","
  //             << this->forward_C3_kernel_global[2] << "\n"
  //             << this->forward_kernel_dims[2] << "\n"
  //             << std::flush;
  status = clEnqueueNDRangeKernel(
      this->cmdQueue, this->forward_C3_kernel, this->forward_kernel_dims[2],
      NULL, this->forward_C3_kernel_global, this->forward_C3_kernel_local, 0,
      NULL, &event);
  cl_event waits[] = {event};
  // clWaitForEvents(1, waits);
  clReleaseEvent(event);
  return true;
}

bool CNN::Forward_S4() {
  cl_int status;
  status = clSetKernelArg(this->forward_S4_kernel, 0, sizeof(cl_mem),
                          &this->device_neuron_C3);
  CHECK_CL(status, "Can't assign args0");
  status |= clSetKernelArg(this->forward_S4_kernel, 1, sizeof(cl_mem),
                           &this->device_weight_S4);
  CHECK_CL(status, "Can't assign args1");
  status |= clSetKernelArg(this->forward_S4_kernel, 2, sizeof(cl_mem),
                           &this->device_bias_S4);
  CHECK_CL(status, "Can't assign args2");
  status |= clSetKernelArg(this->forward_S4_kernel, 3, sizeof(cl_mem),
                           &this->device_neuron_S4);
  CHECK_CL(status, "Can't assign args3");

  cl_event event;
  //   std::cout << "forward S4 global: "
  //   			<< this->forward_S4_kernel_global[0] << ","
  //             << this->forward_S4_kernel_global[1] << ","
  //             << this->forward_S4_kernel_global[2] << "\n"
  //             << this->forward_kernel_dims[3] << "\n"
  //             << std::flush;
  status = clEnqueueNDRangeKernel(
      this->cmdQueue, this->forward_S4_kernel, this->forward_kernel_dims[3],
      NULL, this->forward_S4_kernel_global, this->forward_S4_kernel_local, 0,
      NULL, &event);
  cl_event waits[] = {event};
  // clWaitForEvents(1, waits);
  clReleaseEvent(event);
  return true;
}

bool CNN::Forward_C5() {
  cl_int status;
  status = clSetKernelArg(this->forward_C5_kernel, 0, sizeof(cl_mem),
                          &this->device_neuron_S4);
  CHECK_CL(status, "Can't assign args0");
  status |= clSetKernelArg(this->forward_C5_kernel, 1, sizeof(cl_mem),
                           &this->device_weight_C5);
  CHECK_CL(status, "Can't assign args1");
  status |= clSetKernelArg(this->forward_C5_kernel, 2, sizeof(cl_mem),
                           &this->device_bias_C5);
  CHECK_CL(status, "Can't assign args2");
  status |= clSetKernelArg(this->forward_C5_kernel, 3, sizeof(cl_mem),
                           &this->device_neuron_C5);
  CHECK_CL(status, "Can't assign args3");

  cl_event event;
  //   std::cout << "forward C5 global: "
  //   			<< this->forward_C5_kernel_global[0] << ","
  //             << this->forward_C5_kernel_global[1] << ","
  //             << this->forward_C5_kernel_global[2] << "\n"
  //             << this->forward_kernel_dims[4] << "\n"
  //             << std::flush;
  status = clEnqueueNDRangeKernel(
      this->cmdQueue, this->forward_C5_kernel, this->forward_kernel_dims[4],
      NULL, this->forward_C5_kernel_global, this->forward_C5_kernel_local, 0,
      NULL, &event);
  cl_event waits[] = {event};
  // clWaitForEvents(1, waits);
  clReleaseEvent(event);
  return true;
}

bool CNN::Forward_output() {
  cl_int status;
  status = clSetKernelArg(this->forward_output_kernel, 0, sizeof(cl_mem),
                          &this->device_neuron_C5);
  CHECK_CL(status, "Can't assign args0");
  status |= clSetKernelArg(this->forward_output_kernel, 1, sizeof(cl_mem),
                           &this->device_weight_output);
  CHECK_CL(status, "Can't assign args1");
  status |= clSetKernelArg(this->forward_output_kernel, 2, sizeof(cl_mem),
                           &this->device_bias_output);
  CHECK_CL(status, "Can't assign args2");
  status |= clSetKernelArg(this->forward_output_kernel, 3, sizeof(cl_mem),
                           &this->device_neuron_output);
  CHECK_CL(status, "Can't assign args3");

  cl_event event;
  //   std::cout << "forward output global: "
  //             << this->forward_output_kernel_global[0] << "\n"
  //             << this->forward_kernel_dims[5] << "\n"
  //             << std::flush;
  status = clEnqueueNDRangeKernel(
      this->cmdQueue, this->forward_output_kernel, this->forward_kernel_dims[5],
      NULL, this->forward_output_kernel_global,
      this->forward_output_kernel_local, 0, NULL, &event);
  cl_event waits[] = {event};
  clWaitForEvents(1, waits);
  clReleaseEvent(event);
  return true;
}
#endif
