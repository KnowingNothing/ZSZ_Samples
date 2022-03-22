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
    __global float *pw;
    pw = weight + addr1;
    __global float *pi = in + addr2;
    sum = 0.0;
    __global float *ppw = pw;
    __global float *ppi = pi + y * in_width + x;
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
