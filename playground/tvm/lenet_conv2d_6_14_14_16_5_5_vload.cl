__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict weight, __global float* restrict bias, __global float* restrict output) {
  float output_local[1];
  __local float data_shared[216];
  __local float weight_shared[150];
  output_local[(0)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer_outer_outer < 54; ++ax0_ax1_fused_ax2_fused_outer_outer_outer_outer) {
    data_shared[((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 4) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))))] = data[((((((((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 4) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))) / 36) * 196) + (((int)get_group_id(1)) * 28)) + ((((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 4) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))) % 36) / 6) * 14)) + (((int)get_group_id(0)) * 2)) + ((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 4) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))) % 6)))];
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer < 19; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer) {
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_s = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_s < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) {
      if (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer * 8) + (((int)get_local_id(1)) * 4)) + (((int)get_local_id(0)) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) < 150) {
        if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer * 4) + (((int)get_local_id(1)) * 2)) + ((int)get_local_id(0))) < 75) {
          weight_shared[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer * 8) + (((int)get_local_id(1)) * 4)) + (((int)get_local_id(0)) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s))] = weight[((((((((int)get_group_id(2)) * 150) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_outer * 8)) + (((int)get_local_id(1)) * 4)) + (((int)get_local_id(0)) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_s))];
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
    for (int rr_outer_inner = 0; rr_outer_inner < 5; ++rr_outer_inner) {
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        for (int rs_inner = 0; rs_inner < 5; ++rs_inner) {
          output_local[(0)] = (output_local[(0)] + ((data_shared[(((((((rc_outer_inner * 108) + (rc_inner * 36)) + (((int)get_local_id(1)) * 6)) + (rr_outer_inner * 6)) + ((int)get_local_id(0))) + rs_inner))] * weight_shared[(((((rc_outer_inner * 75) + (rc_inner * 25)) + (rr_outer_inner * 5)) + rs_inner))]) + (bias[(((int)get_group_id(2)))] * 6.666667e-03f)));
        }
      }
    }
  }
  output[((((((((int)get_group_id(2)) * 100) + (((int)get_group_id(1)) * 20)) + (((int)get_local_id(1)) * 10)) + (((int)get_group_id(0)) * 2)) + ((int)get_local_id(0))))] = output_local[(0)];
}


