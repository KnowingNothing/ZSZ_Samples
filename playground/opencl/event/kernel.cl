__kernel void test1(__global float* input1, __global float* input2, __global float* output) {
    int idx = get_global_id(0);
    output[idx] = input1[idx] + input2[idx];
}

__kernel void test2(__global float* input1, __global float* input2, __global float* output) {
    int idx = get_global_id(0);
    output[idx] = output[idx] * input1[idx] - input2[idx];
    // ret
}