__kernel void kernel_backward_S4_weight(__global float *delta_neuron_C5,
                                        __global float *neuron_S4,
                                        __global float *delta_weight_C5,
                                        __global float *delta_bias_C5) {
  __global float *grad_to_output = delta_neuron_C5;
  __global float *input = neuron_S4;
  __global float *grad_to_weight = delta_weight_C5;
  __global float *grad_to_bias = delta_bias_C5;
  for (int r = 0; r < 5; ++r) {
    for (int s = 0; s < 5; ++s) {
      grad_to_weight[((
          (((((((int)get_group_id(1)) * 800) + (((int)get_local_id(1)) * 400)) +
             (((int)get_group_id(0)) * 50)) +
            (((int)get_local_id(0)) * 25)) +
           (r * 5)) +
          s))] = 0.000000e+00f;
      grad_to_weight[((
          (((((((int)get_group_id(1)) * 800) + (((int)get_local_id(1)) * 400)) +
             (((int)get_group_id(0)) * 50)) +
            (((int)get_local_id(0)) * 25)) +
           (r * 5)) +
          s))] = (grad_to_weight[(((((((((int)get_group_id(1)) * 800) +
                                       (((int)get_local_id(1)) * 400)) +
                                      (((int)get_group_id(0)) * 50)) +
                                     (((int)get_local_id(0)) * 25)) +
                                    (r * 5)) +
                                   s))] +
                  (grad_to_output[(((((int)get_group_id(1)) * 2) +
                                    ((int)get_local_id(1))))] *
                   input[(((((((int)get_group_id(0)) * 50) +
                             (((int)get_local_id(0)) * 25)) +
                            (r * 5)) +
                           s))]));
    }
  }
}
