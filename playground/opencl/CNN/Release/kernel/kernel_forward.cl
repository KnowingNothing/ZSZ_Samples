__kernel void kernel_forward_C1(__global float *in, __global float *weight,
                                __global float *bias, __global float *out,
                                int image_offset) {
  int channel = 6;
  int out_width = 28;
  int out_height = 28;
  int kernel_width = 5;
  int kernel_height = 5;
  int in_num = 1;
  int in_width = 32;
  int in_heigh = 32;
  channel = get_global_id(0);
  int y = get_global_id(1);
  int x = get_global_id(2);
  int index = (channel * out_height * out_width) + y * out_width + x;
  float sum = 0.0;
  int inc = 0;
  int wx = 0;
  int wy = 0;
  out[index] = 0.0;
  for (inc = 0; inc < in_num; inc++) {
    int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
    int addr2 = (inc)*in_width * in_height;
    const float *pw = weight + addr1;           //卷积核
    const float *pi = in + image_offset + addr2; //输入图像
    sum = 0.0;
    const float *ppw = pw;
    const float *ppi = pi + y * in_width + x;
    for (wy = 0; wy < kernel_height; wy++) {
      for (wx = 0; wx < kernel_width; wx++) {
        sum += *ppw++ * ppi[wy * in_width + wx];
      }
    }
    out[index] += sum;
  }
  out[index] += bias[channel];
  out[index] = tanh((float)(out[index]));
}

__kernel void kernel_forward_S2(__global float *in, __global float *weight,
                                __global float *bias, __global float *out) {
  int channel = 6;
  int out_width = 14;
  int out_height = 14;
  int kernel_width = 2;
  int kernel_height = 2;
  int in_num = 6;
  int in_width = 28;
  int in_height = 28;
  channel = get_global_id(0);
  int y = get_global_id(1);
  int x = get_global_id(2);
  // float scale_factor = 1.0 / (kernel_width * kernel_height);
  int block = in_width * in_height * channel;
  int rows = y * kernel_width;
  int cols = x * kernel_height;
  int index = (channel * out_height * out_width) + y * out_width + x;

  out[index] = 0.0;
  for (int m = 0; m < kernel_width; m++) {
    for (int n = 0; n < kernel_height; n++) {
      out[index] +=
          weight[channel] * in[(rows + m) * in_width + cols + n + block];
    }
  }
  out[index] *= 0.25; // scale_factor;
  out[index] += bias[channel];
  out[index] = tanh((float)(out[index]));
}

__kernel void kernel_forward_C3(__global float *in, __global float *weight,
                                __global float *bias, __global float *out,
                                __global bool *tbl) {
  int channel = 16;
  int out_width = 10;
  int out_height = 10;
  int kernel_width = 5;
  int kernel_height = 5;
  int in_num = 6;
  int in_width = 14;
  int in_height = 14;
  channel = get_global_id(0);
  int y = get_global_id(1);
  int x = get_global_id(2);
  int index = (channel * out_height * out_width) + y * out_width + x;
  float sum = 0.0;
  int inc = 0;
  int wx = 0;
  int wy = 0;
  out[index] = 0.0;
  for (inc = 0; inc < in_num; inc++) {
    if (!tbl[inc * 16 + channel])
      continue;
    int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
    int addr2 = (inc)*in_width * in_height;
    const float *pw = weight + addr1; //卷积核
    const float *pi = in + addr2;     //输入图像
    sum = 0.0;
    const float *ppw = pw;
    const float *ppi = pi + y * in_width + x;
    for (wy = 0; wy < kernel_height; wy++) {
      for (wx = 0; wx < kernel_width; wx++) {
        sum += *ppw++ * ppi[wy * in_width + wx];
      }
    }
    out[index] += sum;
  }
  out[index] += bias[channel];
  out[index] = tanh((float)(out[index]));
}

__kernel void kernel_forward_S4(__global float *in, __global float *weight,
                                __global float *bias, __global float *out) {
  int channel = 16;
  int out_width = 5;
  int out_height = 5;
  int kernel_width = 2;
  int kernel_height = 2;
  int in_num = 16;
  int in_width = 10;
  int in_height = 10;
  channel = get_global_id(0);
  int y = get_global_id(1);
  int x = get_global_id(2);
  // float scale_factor = 1.0 / (kernel_width * kernel_height);
  int block = in_width * in_height * channel;
  int rows = y * kernel_width;
  int cols = x * kernel_height;
  int index = (channel * out_height * out_width) + y * out_width + x;

  out[index] = 0.0;
  for (int m = 0; m < kernel_width; m++) {
    for (int n = 0; n < kernel_height; n++) {
      out[index] +=
          weight[channel] * in[(rows + m) * in_width + cols + n + block];
    }
  }
  out[index] *= 0.25; // scale_factor;
  out[index] += bias[channel];
  out[index] = tanh((float)(out[index]));
}

__kernel void kernel_forward_C5(__global float *in, __global float *weight,
                                __global float *bias, __global float *out) {
  int channel = 120;
  int out_width = 1;
  int out_height = 1;
  int kernel_width = 5;
  int kernel_height = 5;
  int in_num = 16;
  int in_width = 5;
  int in_height = 5;
  channel = get_global_id(0);
  int y = get_global_id(1);
  int x = get_global_id(2);
  int index = (channel * out_height * out_width) + y * out_width + x;
  float sum = 0.0;
  int inc = 0;
  int wx = 0;
  int wy = 0;
  out[index] = 0.0;
  for (inc = 0; inc < in_num; inc++) {
    int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
    int addr2 = (inc)*in_width * in_height;
    const float *pw = weight + addr1; //卷积核
    const float *pi = in + addr2;     //输入图像
    sum = 0.0;
    const float *ppw = pw;
    const float *ppi = pi + y * in_width + x;
    for (wy = 0; wy < kernel_height; wy++) {
      for (wx = 0; wx < kernel_width; wx++) {
        sum += *ppw++ * ppi[wy * in_width + wx];
      }
    }
    out[index] += sum;
  }
  out[index] += bias[channel];
  out[index] = tanh((float)(out[index]));
}

__kernel void kernel_forward_output(__global float *in, __global float *weight,
                                    __global float *bias, __global float *out) {
    int num_neuron_output_CNN = 10;
    int num_neuron_C5_CNN = 120;
    int x = get_global_id(0);

    out[x] = 0.0;
    for (int c = 0; c < num_neuron_C5_CNN; c++) {
        out[x] += weight[c * num_neuron_output_CNN + x] * in[c];
    }
    out[x] += bias[x];
    out[x] = tanh(out[x]);
}
