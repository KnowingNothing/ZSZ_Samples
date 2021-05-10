__kernel void kernel_backward_S2_weight(__global float *delta_neuron_C3,
                                        __global float *neuron_S2,
                                        __global float *delta_weight_C3,
                                        __global float *delta_bias_C3) {
  __global float *grad_to_output = delta_neuron_C3;
  __global float *input = neuron_S2;
  __global float *grad_to_weight = delta_weight_C3;
  __global float *grad_to_bias = delta_bias_C3;
  for (int r = 0; r < 5; ++r) {
    for (int s = 0; s < 5; ++s) {
      grad_to_weight[((
          (((((((int)get_group_id(1)) * 300) + (((int)get_local_id(1)) * 150)) +
             (((int)get_group_id(0)) * 50)) +
            (((int)get_local_id(0)) * 25)) +
           (r * 5)) +
          s))] = 0.000000e+00f;
      for (int rp = 0; rp < 10; ++rp) {
        for (int rq = 0; rq < 10; ++rq) {
          grad_to_weight[(((((((((int)get_group_id(1)) * 300) +
                               (((int)get_local_id(1)) * 150)) +
                              (((int)get_group_id(0)) * 50)) +
                             (((int)get_local_id(0)) * 25)) +
                            (r * 5)) +
                           s))] =
              (grad_to_weight[(((((((((int)get_group_id(1)) * 300) +
                                    (((int)get_local_id(1)) * 150)) +
                                   (((int)get_group_id(0)) * 50)) +
                                  (((int)get_local_id(0)) * 25)) +
                                 (r * 5)) +
                                s))] +
               (grad_to_output[(((((((int)get_group_id(1)) * 200) +
                                   (((int)get_local_id(1)) * 100)) +
                                  (rp * 10)) +
                                 rq))] *
                input[(((((((((int)get_group_id(0)) * 392) +
                            (((int)get_local_id(0)) * 196)) +
                           (r * 14)) +
                          (rp * 14)) +
                         s) +
                        rq))]));
        }
      }
    }
  }
}
