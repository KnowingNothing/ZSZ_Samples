__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict weight, __global float* restrict bias, __global float* restrict output) {
  float output_local[1];
  __local float data_shared[16];
  __local float weight_shared[1];
  output_local[(0)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_outer_outer_outer_outer) {
    data_shared[((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 4) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))))] = data[(((((((((int)get_group_id(2)) * 784) + (((int)get_group_id(1)) * 112)) + (ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 28)) + (((int)get_group_id(0)) * 4)) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))))];
  }
  if (((((int)get_local_id(1)) * 2) + ((int)get_local_id(0))) < 1) {
    if (((int)get_local_id(1)) < 1) {
      weight_shared[(((((int)get_local_id(1)) * 2) + ((int)get_local_id(0))))] = weight[((((((int)get_local_id(1)) * 2) + ((int)get_local_id(0))) + ((int)get_group_id(2))))];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int rr_outer_inner = 0; rr_outer_inner < 2; ++rr_outer_inner) {
    for (int rs_inner = 0; rs_inner < 2; ++rs_inner) {
      if (((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) < 13) {
        if (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) < 13) {
          output_local[(0)] = (output_local[(0)] + (((data_shared[(((((((int)get_local_id(1)) * 8) + (rr_outer_inner * 4)) + (((int)get_local_id(0)) * 2)) + rs_inner))] * weight_shared[(0)]) * 2.500000e-01f) + (bias[(((int)get_group_id(2)))] * 2.500000e-01f)));
        }
      }
    }
  }
  if (((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) < 13) {
    if (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) < 13) {
      output[((((((((int)get_group_id(2)) * 169) + (((int)get_group_id(1)) * 26)) + (((int)get_local_id(1)) * 13)) + (((int)get_group_id(0)) * 2)) + ((int)get_local_id(0))))] = output_local[(0)];
    }
  }
}


