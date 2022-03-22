__kernel void kernel_backward_C3_input(__global float *delta_neuron_S4,
                                       __global float *neuron_C3,
                                       __global float *weight_S4,
                                       __global float *delta_neuron_C3) {
  __global float *weight = weight_S4;
  __global float *grad_to_output = delta_neuron_S4;
  __global float *grad_to_input = delta_neuron_C3;
  __global float *input = neuron_C3;
  grad_to_input[(
      ((((((((int)get_group_id(2)) * 200) + (((int)get_local_id(2)) * 100)) +
          (((int)get_group_id(1)) * 20)) +
         (((int)get_local_id(1)) * 10)) +
        (((int)get_group_id(0)) * 2)) +
       ((int)get_local_id(0))))] = 0.000000e+00f;
  for (int rr = 0; rr < 2; ++rr) {
    for (int rs = 0; rs < 2; ++rs) {
      grad_to_input[((
          (((((((int)get_group_id(2)) * 200) + (((int)get_local_id(2)) * 100)) +
             (((int)get_group_id(1)) * 20)) +
            (((int)get_local_id(1)) * 10)) +
           (((int)get_group_id(0)) * 2)) +
          ((int)get_local_id(0))))] =
          (grad_to_input[(((((((((int)get_group_id(2)) * 200) +
                               (((int)get_local_id(2)) * 100)) +
                              (((int)get_group_id(1)) * 20)) +
                             (((int)get_local_id(1)) * 10)) +
                            (((int)get_group_id(0)) * 2)) +
                           ((int)get_local_id(0))))] +
           ((((((1 <= (((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) +
                       rr)) &&
                ((((((int)get_group_id(1)) * 2) + ((int)get_local_id(1))) +
                  rr) < 10)) &&
               (1 <=
                (((((int)get_group_id(0)) * 2) + ((int)get_local_id(0))) + rs)))
                  ? grad_to_output[((((((((((((int)get_group_id(2)) * 162) +
                                            (((int)get_local_id(2)) * 81)) +
                                           (((int)get_group_id(1)) * 18)) +
                                          (((int)get_local_id(1)) * 9)) +
                                         (rr * 9)) +
                                        (((int)get_group_id(0)) * 2)) +
                                       ((int)get_local_id(0))) +
                                      rs) -
                                     10))]
                  : 0.000000e+00f) *
             weight[(
                 ((((int)get_group_id(2)) * 2) + ((int)get_local_id(2))))]) *
            (1.000000e+00f - (input[(((((((((int)get_group_id(2)) * 200) +
                                          (((int)get_local_id(2)) * 100)) +
                                         (((int)get_group_id(1)) * 20)) +
                                        (((int)get_local_id(1)) * 10)) +
                                       (((int)get_group_id(0)) * 2)) +
                                      ((int)get_local_id(0))))] *
                              input[(((((((((int)get_group_id(2)) * 200) +
                                          (((int)get_local_id(2)) * 100)) +
                                         (((int)get_group_id(1)) * 20)) +
                                        (((int)get_local_id(1)) * 10)) +
                                       (((int)get_group_id(0)) * 2)) +
                                      ((int)get_local_id(0))))]))));
    }
  }
}
