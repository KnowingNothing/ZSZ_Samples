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
  int in_height = 32;
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
    __global const float *pw = weight + addr1;
    __global const float *pi = in + image_offset + addr2;
    sum = 0.0;
    __global const float *ppw = pw;
    __global const float *ppi = pi + y * in_width + x;
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
