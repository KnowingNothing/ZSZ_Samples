#define activation_function_tanh_derivative(x) (1.0 - x * x)

__kernel void kernel_backward_C5(__global float *delta_neuron_output,
                                 __global float *neuron_C5,
                                 __global float *weight_output,
                                 __global float *delta_neuron_C5,
                                 __global float *delta_weight_output,
                                 __global float *delta_bias_output) {
  // propagate delta to previous layer
  // cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])
  // pre_w_kj_delta[j] += pre_delta[j]*n[k];
  // pre_b_kj_delta[j] += pre_delta[j]
  int num_neuron_output_CNN = 10;
  int len_weight_output_CNN = 120 * 10;
  int len_bias_output_CNN = 10;
  int num_neuron_C5_CNN = 120;
  int k = get_global_id(0);

  delta_neuron_C5[k] = 0.0;
  for (int j = 0; j < num_neuron_output_CNN; j++) {
    int addr1 = k * num_neuron_output_CNN + j; //当前权重
    int addr2 = j;
    delta_neuron_C5[k] += delta_neuron_output[j] * weight_output[addr1] *
                          activation_function_tanh_derivative(neuron_C5[k]);
    delta_weight_output[addr1] = delta_neuron_output[j] * neuron_C5[k];
    delta_bias_output[j] = delta_neuron_output[j];
  }
}
