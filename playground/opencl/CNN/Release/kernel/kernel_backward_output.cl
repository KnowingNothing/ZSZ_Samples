__kernel void kernel_backward_output(__global float *neuron_output,
                                     __global float *label,
                                     __global float *delta_neuron_output,
                                     int label_offset) {
  int num_neuron_output_CNN = 10;
  int i = get_global_id(0);
  float dE_dy;
  float dy_da;
  // loss
  dE_dy = (neuron_output[i] - label[label_offset + i]);
  dy_da = (1.0 - neuron_output[i] * neuron_output[i]);

  // delta = dE/da = (dE/dy) * (dy/da)
  delta_neuron_output[i] = dE_dy * dy_da;
}
