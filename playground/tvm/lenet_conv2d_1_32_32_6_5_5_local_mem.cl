__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict weight, __global float* restrict bias, __global float* restrict output) {
  float output_local[1];
  output_local[(0)] = 0.000000e+00f;
  for (int rr_outer_inner = 0; rr_outer_inner < 5; ++rr_outer_inner) {
    for (int rs_inner = 0; rs_inner < 5; ++rs_inner) {
      output_local[(0)] = (output_local[(0)] + ((data[(((((((((int)get_group_id(1)) * 224) + (((int)get_local_id(1)) * 32)) + (rr_outer_inner * 32)) + (((int)get_group_id(0)) * 7)) + ((int)get_local_id(0))) + rs_inner))] * weight[((((((int)get_group_id(2)) * 25) + (rr_outer_inner * 5)) + rs_inner))]) + (bias[(((int)get_group_id(2)))] * 4.000000e-02f)));
    }
  }
  output[((((((((int)get_group_id(2)) * 784) + (((int)get_group_id(1)) * 196)) + (((int)get_local_id(1)) * 28)) + (((int)get_group_id(0)) * 7)) + ((int)get_local_id(0))))] = output_local[(0)];
}


