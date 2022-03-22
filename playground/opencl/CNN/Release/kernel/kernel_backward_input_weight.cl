__kernel void kernel_backward_input_weight(__global float *delta_neuron_C1,
                                           __global float *data_input_train,
                                           __global float *delta_weight_C1,
                                           __global float *delta_bias_C1,
                                           int image_offset) {
  __global float *grad_to_output = delta_neuron_C1;
  __global float *input = data_input_train + image_offset;
  __global float *grad_to_weight = delta_weight_C1;
  __global float *grad_to_bias = delta_bias_C1;
  for (int r = 0; r < 5; ++r) {
    for (int s = 0; s < 5; ++s) {
      grad_to_weight[((((((int)get_global_id(1)) * 25) + (r * 5)) + s))] =
          0.000000e+00f;
      for (int rp = 0; rp < 28; ++rp) {
        for (int rq = 0; rq < 28; ++rq) {
          grad_to_weight[((((((int)get_global_id(1)) * 25) + (r * 5)) + s))] =
              (grad_to_weight[(
                   (((((int)get_global_id(1)) * 25) + (r * 5)) + s))] +
               (grad_to_output[(
                    (((((int)get_global_id(1)) * 784) + (rp * 28)) + rq))] *
                input[(((((r * 32) + (rp * 32)) + s) + rq))]));
        }
      }
    }
  }
  grad_to_bias[get_global_id(1)] = grad_to_output[get_global_id(1)];
}
