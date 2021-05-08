__kernel void kernel_update(__global float *delta, __global float *error,
                            __global float *weight, float lr, float eps,
                            int len) {
  int i = get_global_id(0);
  if (i < len) {
    error[i] = delta[i] * delta[i];
    weight[i] -= lr * delta[i] / (sqrt(error[i]) + eps);
  }
}
