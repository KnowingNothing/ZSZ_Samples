__kernel void kernel_backward_S2_input(__global float *delta_neuron_C3,
                                       __global float *neuron_S2,
                                       __global float *weight_C3,
                                       __global float *delta_neuron_S2,
                                       __global bool *tbl) {
  __global float *weight = weight_C3;
  __global float *grad_to_output = delta_neuron_C3;
  __global float *grad_to_input = delta_neuron_S2;
  __global float *input = neuron_S2;
  grad_to_input[(
      ((((((((int)get_group_id(2)) * 392) + (((int)get_local_id(2)) * 196)) +
          (((int)get_group_id(1)) * 28)) +
         (((int)get_local_id(1)) * 14)) +
        (((int)get_group_id(0)) * 2)) +
       ((int)get_local_id(0))))] = 0.000000e+00f;
  for (int rk = 0; rk < 16; ++rk) {
    for (int rr = 0; rr < 5; ++rr) {
      for (int rs = 0; rs < 5; ++rs) {
        grad_to_input[(((((((((int)get_group_id(2)) * 392) +
                            (((int)get_local_id(2)) * 196)) +
                           (((int)get_group_id(1)) * 28)) +
                          (((int)get_local_id(1)) * 14)) +
                         (((int)get_group_id(0)) * 2)) +
                        ((int)get_local_id(0))))] =
            (grad_to_input[(((((((((int)get_group_id(2)) * 392) +
                                 (((int)get_local_id(2)) * 196)) +
                                (((int)get_group_id(1)) * 28)) +
                               (((int)get_local_id(1)) * 14)) +
                              (((int)get_group_id(0)) * 2)) +
                             ((int)get_local_id(0))))] +
             ((((((4 <=
                   (((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) +
                    rr)) &&
                  ((((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) +
                    rr) < 14)) &&
                 (4 <=
                  (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) +
                   rs)))
                    ? grad_to_output[(
                          ((((((((rk * 100) + (((int)get_group_id(1)) * 20)) +
                                (((int)get_local_id(1)) * 10)) +
                               (rr * 10)) +
                              (((int)get_group_id(0)) * 2)) +
                             ((int)get_local_id(0))) +
                            rs) -
                           44))]
                    : 0.000000e+00f) *
               weight[(((((((rk * 150) + (((int)get_group_id(2)) * 50)) +
                           (((int)get_local_id(2)) * 25)) +
                          36) -
                         rs) -
                        (rr * 5)))]) *
              (1.000000e+00f - (input[(((((((((int)get_group_id(2)) * 392) +
                                            (((int)get_local_id(2)) * 196)) +
                                           (((int)get_group_id(1)) * 28)) +
                                          (((int)get_local_id(1)) * 14)) +
                                         (((int)get_group_id(0)) * 2)) +
                                        ((int)get_local_id(0))))] *
                                input[(((((((((int)get_group_id(2)) * 392) +
                                            (((int)get_local_id(2)) * 196)) +
                                           (((int)get_group_id(1)) * 28)) +
                                          (((int)get_local_id(1)) * 14)) +
                                         (((int)get_group_id(0)) * 2)) +
                                        ((int)get_local_id(0))))]))));
      }
    }
  }
}
