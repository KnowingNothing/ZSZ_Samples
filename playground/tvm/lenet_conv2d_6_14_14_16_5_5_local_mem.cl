__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict weight, __global float* restrict bias, __global float* restrict output) {
  float output_local[1];
  output_local[(0)] = 0.000000e+00f;
  for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
    for (int rr_outer_inner = 0; rr_outer_inner < 5; ++rr_outer_inner) {
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        for (int rs_inner = 0; rs_inner < 5; ++rs_inner) {
          output_local[(0)] = (output_local[(0)] + ((data[(((((((((rc_outer_inner * 588) + (rc_inner * 196)) + (((int)get_group_id(1)) * 28)) + (((int)get_local_id(1)) * 14)) + (rr_outer_inner * 14)) + (((int)get_group_id(0)) * 2)) + ((int)get_local_id(0))) + rs_inner))] * weight[((((((((int)get_group_id(2)) * 150) + (rc_outer_inner * 75)) + (rc_inner * 25)) + (rr_outer_inner * 5)) + rs_inner))]) + (bias[(((int)get_group_id(2)))] * 6.666667e-03f)));
        }
      }
    }
  }
  output[((((((((int)get_group_id(2)) * 100) + (((int)get_group_id(1)) * 20)) + (((int)get_local_id(1)) * 10)) + (((int)get_group_id(0)) * 2)) + ((int)get_local_id(0))))] = output_local[(0)];
}


