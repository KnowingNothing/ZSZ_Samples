__kernel void default_function_kernel0(__global float *restrict data,
                                       __global float *restrict weight,
                                       __global float *restrict bias,
                                       __global float *restrict output) {
  float output_local[1];
  __local float data_shared[121];
  __local float weight_shared[25];
  output_local[(0)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer_outer_outer = 0;
       ax0_ax1_fused_ax2_fused_outer_outer_outer_outer < 3;
       ++ax0_ax1_fused_ax2_fused_outer_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 49) +
          (((int)get_local_id(1)) * 7)) +
         ((int)get_local_id(0))) < 121) {
      if (((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 7) +
           ((int)get_local_id(1))) < 18) {
        data_shared[((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 49) +
                       (((int)get_local_id(1)) * 7)) +
                      ((int)get_local_id(0))))] =
            data[(
                ((((((int)get_group_id(1)) * 224) +
                   (((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 49) +
                       (((int)get_local_id(1)) * 7)) +
                      ((int)get_local_id(0))) /
                     11) *
                    32)) +
                  (((int)get_group_id(0)) * 7)) +
                 ((((ax0_ax1_fused_ax2_fused_outer_outer_outer_outer * 49) +
                    (((int)get_local_id(1)) * 7)) +
                   ((int)get_local_id(0))) %
                  11)))];
      }
    }
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_s = 0;
       ax0_ax1_fused_ax2_fused_ax3_fused_inner_s < 2;
       ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) {
    if ((((((int)get_local_id(1)) * 14) + (((int)get_local_id(0)) * 2)) +
         ax0_ax1_fused_ax2_fused_ax3_fused_inner_s) < 25) {
      if (((((int)get_local_id(1)) * 7) + ((int)get_local_id(0))) < 13) {
        if (((int)get_local_id(1)) < 2) {
          weight_shared[(
              (((((int)get_local_id(1)) * 14) + (((int)get_local_id(0)) * 2)) +
               ax0_ax1_fused_ax2_fused_ax3_fused_inner_s))] =
              weight[(((((((int)get_group_id(2)) * 25) +
                         (((int)get_local_id(1)) * 14)) +
                        (((int)get_local_id(0)) * 2)) +
                       ax0_ax1_fused_ax2_fused_ax3_fused_inner_s))];
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int rr_outer_inner = 0; rr_outer_inner < 5; ++rr_outer_inner) {
    for (int rs_inner = 0; rs_inner < 5; ++rs_inner) {
      output_local[(0)] =
          (output_local[(0)] +
           ((data_shared[(
                 ((((((int)get_local_id(1)) * 11) + (rr_outer_inner * 11)) +
                   ((int)get_local_id(0))) +
                  rs_inner))] *
             weight_shared[(((rr_outer_inner * 5) + rs_inner))]) +
            (bias[(((int)get_group_id(2)))] * 4.000000e-02f)));
    }
  }
  output[((((((((int)get_group_id(2)) * 784) + (((int)get_group_id(1)) * 196)) +
             (((int)get_local_id(1)) * 28)) +
            (((int)get_group_id(0)) * 7)) +
           ((int)get_local_id(0))))] = output_local[(0)];
}
