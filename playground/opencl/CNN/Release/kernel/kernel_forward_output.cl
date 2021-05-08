__kernel void kernel_forward_output(__global float *in, __global float *weight,
                                    __global float *bias, __global float *out) {
    int num_neuron_output_CNN = 10;
    int num_neuron_C5_CNN = 120;
    int x = get_global_id(0);

    out[x] = 0.0;
    for (int c = 0; c < num_neuron_C5_CNN; c++) {
        out[x] += weight[c * num_neuron_output_CNN + x] * in[c];
    }
    out[x] += bias[x];
    out[x] = tanh(out[x]);
}
