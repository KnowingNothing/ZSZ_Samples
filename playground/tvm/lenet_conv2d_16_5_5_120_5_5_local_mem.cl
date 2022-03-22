__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict weight, __global float* restrict bias, __global float* restrict output) {
  float output_local[1];
  output_local[(0)] = 0.000000e+00f;
  for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
    for (int rr_outer_inner = 0; rr_outer_inner < 5; ++rr_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int rs_inner = 0; rs_inner < 5; ++rs_inner) {
          output_local[(0)] = (output_local[(0)] + ((data[(((((rc_outer_inner * 100) + (rc_inner * 25)) + (rr_outer_inner * 5)) + rs_inner))] * weight[(((((((((int)get_group_id(2)) * 800) + (((int)get_local_id(2)) * 400)) + (rc_outer_inner * 100)) + (rc_inner * 25)) + (rr_outer_inner * 5)) + rs_inner))]) + (bias[(((((int)get_group_id(2)) * 2) + ((int)get_local_id(2))))] * 2.500000e-03f)));
        }
      }
    }
  }
  output[(((((int)get_group_id(2)) * 2) + ((int)get_local_id(2))))] = output_local[(0)];
}


