__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict weight, __global float* restrict bias, __global float* restrict output) {
  float output_local[1];
  output_local[(0)] = 0.000000e+00f;
  for (int rr_outer_inner = 0; rr_outer_inner < 2; ++rr_outer_inner) {
    for (int rs_inner = 0; rs_inner < 2; ++rs_inner) {
      output_local[(0)] = (output_local[(0)] + (((data[((((((((int)get_group_id(2)) * 100) + (((int)get_group_id(1)) * 20)) + (rr_outer_inner * 10)) + (((int)get_group_id(0)) * 2)) + rs_inner))] * weight[(((int)get_group_id(2)))]) * 2.500000e-01f) + (bias[(((int)get_group_id(2)))] * 2.500000e-01f)));
    }
  }
  output[((((((int)get_group_id(2)) * 16) + (((int)get_group_id(1)) * 4)) + ((int)get_group_id(0))))] = output_local[(0)];
}


