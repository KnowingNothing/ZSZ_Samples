__kernel void kernel_backward_C1_weight(__global float *delta_neuron_S2,
                                        __global float *neuron_C1,
                                        __global float *delta_weight_S2,
                                        __global float *delta_bias_S2) {
  __global float *grad_to_output = delta_neuron_S2;
  __global float *input = neuron_C1;
  __global float *grad_to_weight = delta_weight_S2;
  __global float *grad_to_bias = delta_bias_S2;
  grad_to_weight[(((int)get_global_id(0)))] = 0.000000e+00f;
  for (int rr = 0; rr < 2; ++rr) {
    for (int rs = 0; rs < 2; ++rs) {
      for (int rp = 0; rp < 27; ++rp) {
        for (int rq = 0; rq < 27; ++rq) {
          grad_to_weight[(((int)get_global_id(0)))] =
              (grad_to_weight[(((int)get_global_id(0)))] +
               (grad_to_output[(
                    (((((int)get_global_id(0)) * 729) + (rp * 27)) + rq))] *
                input[((((((((int)get_global_id(0)) * 784) + (rr * 28)) +
                          (rp * 28)) +
                         rs) +
                        rq))]));
        }
      }
    }
  }
  grad_to_bias[get_global_id(0)] = grad_to_output[get_global_id(0)];
}
