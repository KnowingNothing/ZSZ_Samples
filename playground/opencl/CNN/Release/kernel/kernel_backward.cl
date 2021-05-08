#define activation_function_tanh_derivative(x) (1.0 - x * x)

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

__kernel void kernel_backward_S4_input(__global float *delta_neuron_C5,
                                       __global float *neuron_S4,
                                       __global float *weight_C5,
                                       __global float *delta_neuron_S4) {
  float *weight = weight_C5;
  float *grad_to_output = delta_neuron_C5;
  float *grad_to_input = delta_neuron_S4;
  float *input = neuron_S4;
  grad_to_input[(
      (((((int)get_global_id(2)) * 25) + (((int)get_global_id(1)) * 5)) +
       ((int)get_global_id(0))))] = 0.000000e+00f;
  for (int rk = 0; rk < 120; ++rk) {
    for (int rr = 0; rr < 5; ++rr) {
      for (int rs = 0; rs < 5; ++rs) {
        grad_to_input[(
            (((((int)get_global_id(2)) * 25) + (((int)get_global_id(1)) * 5)) +
             ((int)get_global_id(0))))] =
            (grad_to_input[((((((int)get_global_id(2)) * 25) +
                              (((int)get_global_id(1)) * 5)) +
                             ((int)get_global_id(0))))] +
             ((((((4 <= (((int)get_global_id(1)) + rr)) &&
                  ((((int)get_global_id(1)) + rr) < 5)) &&
                 (4 <= (((int)get_global_id(0)) + rs)))
                    ? grad_to_output[((((((rk + ((int)get_global_id(1))) + rr) +
                                         ((int)get_global_id(0))) +
                                        rs) -
                                       8))]
                    : 0.000000e+00f) *
               weight[(
                   (((((rk * 400) + (((int)get_global_id(2)) * 25)) + 36) - rs) -
                    (rr * 5)))]) *
              (1.000000e+00f - (input[((((((int)get_global_id(2)) * 25) +
                                         (((int)get_global_id(1)) * 5)) +
                                        ((int)get_global_id(0))))] *
                                input[((((((int)get_global_id(2)) * 25) +
                                         (((int)get_global_id(1)) * 5)) +
                                        ((int)get_global_id(0))))]))));
      }
    }
  }
}

__kernel void kernel_backward_S4_weight(__global float *delta_neuron_C5,
                                        __global float *neuron_S4,
                                        __global float *delta_weight_C5,
                                        __global float *delta_bias_C5) {
  float *grad_to_output = delta_neuron_C5;
  float *input = neuron_S4;
  float *grad_to_weight = delta_weight_C5;
  float *grad_to_bias = delta_bias_C5;
  for (int r = 0; r < 5; ++r) {
    for (int s = 0; s < 5; ++s) {
      grad_to_weight[(
          ((((((int)get_global_id(1)) * 400) + (((int)get_global_id(0)) * 25)) +
            (r * 5)) +
           s))] = 0.000000e+00f;
      grad_to_weight[(
          ((((((int)get_global_id(1)) * 400) + (((int)get_global_id(0)) * 25)) +
            (r * 5)) +
           s))] = (grad_to_weight[(((((((int)get_global_id(1)) * 400) +
                                      (((int)get_global_id(0)) * 25)) +
                                     (r * 5)) +
                                    s))] +
                   (grad_to_output[(((int)get_global_id(1)))] *
                    input[((((((int)get_global_id(0)) * 25) + (r * 5)) + s))]));
    }
  }
  grad_to_bias[get_global_id(1)] = grad_to_output[get_global_id(1)];
}

__kernel void kernel_backward_C3_input(__global float *delta_neuron_S4,
                                       __global float *neuron_C3,
                                       __global float *weight_S4,
                                       __global float *delta_neuron_C3) {
  float *weight = weight_S4;
  float *grad_to_output = delta_neuron_S4;
  float *grad_to_input = delta_neuron_C3;
  float *input = neuron_C3;
  grad_to_input[(
      (((((int)get_global_id(2)) * 100) + (((int)get_global_id(1)) * 10)) +
       ((int)get_global_id(0))))] = 0.000000e+00f;
  for (int rr = 0; rr < 2; ++rr) {
    for (int rs = 0; rs < 2; ++rs) {
      grad_to_input[(
          (((((int)get_global_id(2)) * 100) + (((int)get_global_id(1)) * 10)) +
           ((int)get_global_id(0))))] =
          (grad_to_input[((((((int)get_global_id(2)) * 100) +
                            (((int)get_global_id(1)) * 10)) +
                           ((int)get_global_id(0))))] +
           ((((((1 <= (((int)get_global_id(1)) + rr)) &&
                ((((int)get_global_id(1)) + rr) < 10)) &&
               (1 <= (((int)get_global_id(0)) + rs)))
                  ? grad_to_output[(((((((((int)get_global_id(2)) * 81) +
                                         (((int)get_global_id(1)) * 9)) +
                                        (rr * 9)) +
                                       ((int)get_global_id(0))) +
                                      rs) -
                                     10))]
                  : 0.000000e+00f) *
             weight[(((int)get_global_id(2)))]) *
            (1.000000e+00f - (input[((((((int)get_global_id(2)) * 100) +
                                       (((int)get_global_id(1)) * 10)) +
                                      ((int)get_global_id(0))))] *
                              input[((((((int)get_global_id(2)) * 100) +
                                       (((int)get_global_id(1)) * 10)) +
                                      ((int)get_global_id(0))))]))));
    }
  }
}

__kernel void kernel_backward_C3_weight(__global float *delta_neuron_S4,
                                        __global float *neuron_C3,
                                        __global float *delta_weight_S4,
                                        __global float *delta_bias_S4) {
  float *grad_to_output = delta_neuron_S4;
  float *input = neuron_C3;
  float *grad_to_weight = delta_weight_S4;
  float *grad_to_bias = delta_bias_S4;
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

__kernel void kernel_backward_S2_input(__global float *delta_neuron_C3,
                                       __global float *neuron_S2,
                                       __global float *weight_C3,
                                       __global float *delta_neuron_S2,
                                       __global bool *tbl) {
  float *weight = weight_C3;
  float *grad_to_output = delta_neuron_C3;
  float *grad_to_input = delta_neuron_S2;
  float *input = neuron_S2;
  grad_to_input[(
      (((((int)get_global_id(2)) * 196) + (((int)get_global_id(1)) * 14)) +
       ((int)get_global_id(0))))] = 0.000000e+00f;
  for (int rk = 0; rk < 16; ++rk) {
    for (int rr = 0; rr < 5; ++rr) {
      for (int rs = 0; rs < 5; ++rs) {
        grad_to_input[(
            (((((int)get_global_id(2)) * 196) + (((int)get_global_id(1)) * 14)) +
             ((int)get_global_id(0))))] =
            (grad_to_input[((((((int)get_global_id(2)) * 196) +
                              (((int)get_global_id(1)) * 14)) +
                             ((int)get_global_id(0))))] +
             ((((((4 <= (((int)get_global_id(1)) + rr)) &&
                  ((((int)get_global_id(1)) + rr) < 14)) &&
                 (4 <= (((int)get_global_id(0)) + rs)))
                    ? grad_to_output[(
                          ((((((rk * 100) + (((int)get_global_id(1)) * 10)) +
                              (rr * 10)) +
                             ((int)get_global_id(0))) +
                            rs) -
                           44))]
                    : 0.000000e+00f) *
               weight[(
                   (((((rk * 150) + (((int)get_global_id(2)) * 25)) + 36) - rs) -
                    (rr * 5)))]) *
              (1.000000e+00f - (input[((((((int)get_global_id(2)) * 196) +
                                         (((int)get_global_id(1)) * 14)) +
                                        ((int)get_global_id(0))))] *
                                input[((((((int)get_global_id(2)) * 196) +
                                         (((int)get_global_id(1)) * 14)) +
                                        ((int)get_global_id(0))))]))));
      }
    }
  }
}

__kernel void kernel_backward_S2_weight(__global float *delta_neuron_C3,
                                        __global float *neuron_S2,
                                        __global float *delta_weight_C3,
                                        __global float *delta_bias_C3) {
  float *grad_to_output = delta_neuron_C3;
  float *input = neuron_S2;
  float *grad_to_weight = delta_weight_C3;
  float *grad_to_bias = delta_bias_C3;
  for (int r = 0; r < 5; ++r) {
    for (int s = 0; s < 5; ++s) {
      grad_to_weight[(
          ((((((int)get_global_id(1)) * 150) + (((int)get_global_id(0)) * 25)) +
            (r * 5)) +
           s))] = 0.000000e+00f;
      for (int rp = 0; rp < 10; ++rp) {
        for (int rq = 0; rq < 10; ++rq) {
          grad_to_weight[(((((((int)get_global_id(1)) * 150) +
                             (((int)get_global_id(0)) * 25)) +
                            (r * 5)) +
                           s))] =
              (grad_to_weight[(((((((int)get_global_id(1)) * 150) +
                                  (((int)get_global_id(0)) * 25)) +
                                 (r * 5)) +
                                s))] +
               (grad_to_output[(
                    (((((int)get_global_id(1)) * 100) + (rp * 10)) + rq))] *
                input[((
                    ((((((int)get_global_id(0)) * 196) + (r * 14)) + (rp * 14)) +
                     s) +
                    rq))]));
        }
      }
    }
  }
  grad_to_bias[get_global_id(1)] = grad_to_output[get_global_id(1)];
}

__kernel void kernel_backward_C1_input(__global float *delta_neuron_S2,
                                       __global float *neuron_C1,
                                       __global float *weight_S2,
                                       __global float *delta_neuron_C1) {
  float *weight = weight_S2;
  float *grad_to_output = delta_neuron_S2;
  float *grad_to_input = delta_neuron_C1;
  float *input = neuron_C1;
  grad_to_input[(
      (((((int)get_global_id(2)) * 784) + (((int)get_global_id(1)) * 28)) +
       ((int)get_global_id(0))))] = 0.000000e+00f;
  for (int rr = 0; rr < 2; ++rr) {
    for (int rs = 0; rs < 2; ++rs) {
      grad_to_input[(
          (((((int)get_global_id(2)) * 784) + (((int)get_global_id(1)) * 28)) +
           ((int)get_global_id(0))))] =
          (grad_to_input[((((((int)get_global_id(2)) * 784) +
                            (((int)get_global_id(1)) * 28)) +
                           ((int)get_global_id(0))))] +
           ((((((1 <= (((int)get_global_id(1)) + rr)) &&
                ((((int)get_global_id(1)) + rr) < 28)) &&
               (1 <= (((int)get_global_id(0)) + rs)))
                  ? grad_to_output[(((((((((int)get_global_id(2)) * 729) +
                                         (((int)get_global_id(1)) * 27)) +
                                        (rr * 27)) +
                                       ((int)get_global_id(0))) +
                                      rs) -
                                     28))]
                  : 0.000000e+00f) *
             weight[(((int)get_global_id(2)))]) *
            (1.000000e+00f - (input[((((((int)get_global_id(2)) * 784) +
                                       (((int)get_global_id(1)) * 28)) +
                                      ((int)get_global_id(0))))] *
                              input[((((((int)get_global_id(2)) * 784) +
                                       (((int)get_global_id(1)) * 28)) +
                                      ((int)get_global_id(0))))]))));
    }
  }
}

__kernel void kernel_backward_C1_weight(__global float *delta_neuron_S2,
                                        __global float *neuron_C1,
                                        __global float *delta_weight_S2,
                                        __global float *delta_bias_S2) {
  float *grad_to_output = delta_neuron_S2;
  float *input = neuron_C1;
  float *grad_to_weight = delta_weight_S2;
  float *grad_to_bias = delta_bias_S2;
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

__kernel void kernel_backward_input_weight(__global float *delta_neuron_C1,
                                           __global float *data_input_train,
                                           __global float *delta_weight_C1,
                                           __global float *delta_bias_C1,
                                           int image_offset) {
  float *grad_to_output = delta_neuron_C1;
  float *input = data_input_train + image_offset;
  float *grad_to_weight = delta_weight_C1;
  float *grad_to_bias = delta_bias_C1;
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
