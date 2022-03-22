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
