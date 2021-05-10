__kernel void kernel_forward_C5(__global float *in, __global float *weight,
                                __global float *bias, __global float *out) {
  __global float* data = in;
  __global float* output = out;
  float output_local[1];
  __local float data_shared[400];
  __local float weight_shared[800];
  output_local[(0)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer_outer = 0;
       ax0_ax1_fused_ax2_fused_outer_outer_outer < 200;
       ++ax0_ax1_fused_ax2_fused_outer_outer_outer) {
    data_shared[(((ax0_ax1_fused_ax2_fused_outer_outer_outer * 2) +
                  ((int)get_local_id(2))))] =
        data[(((ax0_ax1_fused_ax2_fused_outer_outer_outer * 2) +
               ((int)get_local_id(2))))];
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0;
       ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 400;
       ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    weight_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 2) +
                    ((int)get_local_id(2))))] =
        weight[((((((int)get_group_id(2)) * 800) +
                  (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 2)) +
                 ((int)get_local_id(2))))];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
    for (int rr_outer_inner = 0; rr_outer_inner < 5; ++rr_outer_inner) {
      for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
        for (int rs_inner = 0; rs_inner < 5; ++rs_inner) {
          output_local[(0)] =
              (output_local[(0)] +
               ((data_shared[(((((rc_outer_inner * 100) + (rc_inner * 25)) +
                                (rr_outer_inner * 5)) +
                               rs_inner))] *
                 weight_shared[((((((((int)get_local_id(2)) * 400) +
                                    (rc_outer_inner * 100)) +
                                   (rc_inner * 25)) +
                                  (rr_outer_inner * 5)) +
                                 rs_inner))]) +
                (0 *
                 2.500000e-03f)));
        }
      }
    }
  }
  output[(((((int)get_group_id(2)) * 2) + ((int)get_local_id(2))))] =
      tanh(output_local[(0)] + bias[(
                     ((((int)get_group_id(2)) * 2) + ((int)get_local_id(2))))]);
}
