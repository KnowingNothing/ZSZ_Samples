__kernel void kernel_forward_S4(__global float *in, __global float *weight,
                                __global float *bias, __global float *out) {
  float output_local[1];
  __local float data_shared[4];
  __local float weight_shared[1];
  output_local[(0)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer_outer_outer = 0;
       ax0_ax1_fused_ax2_fused_outer_outer_outer_outer < 4;
       ++ax0_ax1_fused_ax2_fused_outer_outer_outer_outer) {
    data_shared[(ax0_ax1_fused_ax2_fused_outer_outer_outer_outer)] = data[(
        (((((((int)get_group_id(2)) * 100) + (((int)get_group_id(1)) * 20)) +
           ((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer >> 1) * 10)) +
          (((int)get_group_id(0)) * 2)) +
         (ax0_ax1_fused_ax2_fused_outer_outer_outer_outer & 1)))];
  }
  weight_shared[(0)] = weight[(((int)get_group_id(2)))];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int rr_outer_inner = 0; rr_outer_inner < 2; ++rr_outer_inner) {
    for (int rs_inner = 0; rs_inner < 2; ++rs_inner) {
      output_local[(0)] = (output_local[(0)] +
                           (((data_shared[(((rr_outer_inner * 2) + rs_inner))] *
                              weight_shared[(0)]) *
                             2.500000e-01f) +
                            (bias[(((int)get_group_id(2)))] * 2.500000e-01f)));
    }
  }
  output[((((((int)get_group_id(2)) * 16) + (((int)get_group_id(1)) * 4)) +
           ((int)get_group_id(0))))] = output_local[(0)];
}
