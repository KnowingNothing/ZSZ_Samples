__kernel void kernel_backward_S4_input(__global float *delta_neuron_C5,
                                       __global float *neuron_S4,
                                       __global float *weight_C5,
                                       __global float *delta_neuron_S4) {
  __global float *weight = weight_C5;
  __global float *grad_to_output = delta_neuron_C5;
  __global float *grad_to_input = delta_neuron_S4;
  __global float *input = neuron_S4;
  if (((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) < 5) {
    if (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) < 5) {
      grad_to_input[(
          ((((((((int)get_group_id(2)) * 50) + (((int)get_local_id(2)) * 25)) +
              (((int)get_group_id(1)) * 10)) +
             (((int)get_local_id(1)) * 5)) +
            (((int)get_group_id(0)) * 2)) +
           ((int)get_local_id(0))))] = 0.000000e+00f;
    }
  }
  for (int rk = 0; rk < 120; ++rk) {
    for (int rr = 0; rr < 5; ++rr) {
      for (int rs = 0; rs < 5; ++rs) {
        if (((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) < 5) {
          if (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) < 5) {
            grad_to_input[(((((((((int)get_group_id(2)) * 50) +
                                (((int)get_local_id(2)) * 25)) +
                               (((int)get_group_id(1)) * 10)) +
                              (((int)get_local_id(1)) * 5)) +
                             (((int)get_group_id(0)) * 2)) +
                            ((int)get_local_id(0))))] =
                (grad_to_input[(((((((((int)get_group_id(2)) * 50) +
                                     (((int)get_local_id(2)) * 25)) +
                                    (((int)get_group_id(1)) * 10)) +
                                   (((int)get_local_id(1)) * 5)) +
                                  (((int)get_group_id(0)) * 2)) +
                                 ((int)get_local_id(0))))] +
                 ((((((4 <= (((((int)get_group_id(1)) * 2) +
                              ((int)get_local_id(1))) +
                             rr)) &&
                      ((((((int)get_group_id(1)) * 2) +
                         ((int)get_local_id(1))) +
                        rr) < 5)) &&
                     (4 <=
                      (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) +
                       rs)))
                        ? grad_to_output[(((((((((((int)get_group_id(1)) * 2) +
                                                 (((int)get_group_id(0)) * 2)) +
                                                rk) +
                                               ((int)get_local_id(1))) +
                                              rr) +
                                             ((int)get_local_id(0))) +
                                            rs) -
                                           8))]
                        : 0.000000e+00f) *
                   weight[(((((((rk * 400) + (((int)get_group_id(2)) * 50)) +
                               (((int)get_local_id(2)) * 25)) +
                              36) -
                             rs) -
                            (rr * 5)))]) *
                  (1.000000e+00f - (input[(((((((((int)get_group_id(2)) * 50) +
                                                (((int)get_local_id(2)) * 25)) +
                                               (((int)get_group_id(1)) * 10)) +
                                              (((int)get_local_id(1)) * 5)) +
                                             (((int)get_group_id(0)) * 2)) +
                                            ((int)get_local_id(0))))] *
                                    input[(((((((((int)get_group_id(2)) * 50) +
                                                (((int)get_local_id(2)) * 25)) +
                                               (((int)get_group_id(1)) * 10)) +
                                              (((int)get_local_id(1)) * 5)) +
                                             (((int)get_group_id(0)) * 2)) +
                                            ((int)get_local_id(0))))]))));
          }
        }
      }
    }
  }
}
