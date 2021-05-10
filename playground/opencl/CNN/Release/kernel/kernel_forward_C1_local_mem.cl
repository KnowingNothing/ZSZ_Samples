__kernel void kernel_forward_C1(__global float *in, __global float *weight,
                                __global float *bias, __global float *out,
                                int image_offset) {
  __global float* data = in + image_offset;
  __global float* output = out;
  float output_local[1];
  __local float data_shared[121];
  __local float weight_shared[25];
  output_local[(0)] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer_outer = 0;
       ax0_ax1_fused_ax2_fused_outer_outer_outer < 3;
       ++ax0_ax1_fused_ax2_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_outer_outer_outer * 49) +
          (((int)get_local_id(1)) * 7)) +
         ((int)get_local_id(0))) < 121) {
      if (((ax0_ax1_fused_ax2_fused_outer_outer_outer * 7) +
           ((int)get_local_id(1))) < 18) {
        data_shared[((((ax0_ax1_fused_ax2_fused_outer_outer_outer * 49) +
                       (((int)get_local_id(1)) * 7)) +
                      ((int)get_local_id(0))))] =
            data[(((((((int)get_group_id(1)) * 224) +
                     (((((ax0_ax1_fused_ax2_fused_outer_outer_outer * 49) +
                         (((int)get_local_id(1)) * 7)) +
                        ((int)get_local_id(0))) /
                       11) *
                      32)) +
                    (((int)get_group_id(0)) * 7)) +
                   ((((ax0_ax1_fused_ax2_fused_outer_outer_outer * 49) +
                      (((int)get_local_id(1)) * 7)) +
                     ((int)get_local_id(0))) %
                    11)))];
      }
    }
  }
  if (((((int)get_local_id(1)) * 7) + ((int)get_local_id(0))) < 25) {
    if (((int)get_local_id(1)) < 4) {
      weight_shared[(((((int)get_local_id(1)) * 7) + ((int)get_local_id(0))))] =
          weight[(
              (((((int)get_group_id(2)) * 25) + (((int)get_local_id(1)) * 7)) +
               ((int)get_local_id(0))))];
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
            (0* 4.000000e-02f)));
    }
  }
  output[((((((((int)get_group_id(2)) * 784) + (((int)get_group_id(1)) * 196)) +
             (((int)get_local_id(1)) * 28)) +
            (((int)get_group_id(0)) * 7)) +
           ((int)get_local_id(0))))] = tanh(output_local[(0)] + bias[(((int)get_group_id(2)))] );
}

