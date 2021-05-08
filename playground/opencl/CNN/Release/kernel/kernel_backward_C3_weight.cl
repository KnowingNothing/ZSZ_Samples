__kernel void kernel_backward_C3_weight(__global float *delta_neuron_S4,
                                        __global float *neuron_C3,
                                        __global float *delta_weight_S4,
                                        __global float *delta_bias_S4) {
  __global float *grad_to_output = delta_neuron_S4;
  __global float *input = neuron_C3;
  __global float *grad_to_weight = delta_weight_S4;
  __global float *grad_to_bias = delta_bias_S4;
  grad_to_weight[(((int)get_global_id(0)))] = 0.000000e+00f;
  for (int rr = 0; rr < 2; ++rr) {
    for (int rs = 0; rs < 2; ++rs) {
      for (int rp = 0; rp < 9; ++rp) {
        for (int rq = 0; rq < 9; ++rq) {
          grad_to_weight[(((int)get_global_id(0)))] =
              (grad_to_weight[(((int)get_global_id(0)))] +
               (grad_to_output[(
                    (((((int)get_global_id(0)) * 81) + (rp * 9)) + rq))] *
                input[((((((((int)get_global_id(0)) * 100) + (rr * 10)) +
                          (rp * 10)) +
                         rs) +
                        rq))]));
        }
      }
    }
  }
  grad_to_bias[get_global_id(0)] = grad_to_output[get_global_id(0)];
}
