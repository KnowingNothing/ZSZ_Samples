
__kernel void matmul_naive(
    __global float *A,
    __global float *B,
    __global float *C,
    int stA, int stB, int stC) {
    
    int global_idx = get_global_id(0);
    int global_idy = get_global_id(1);

    float accum = 0;
    for (int k = 0; k < stA; ++k) {
        accum += A[global_idx * stA + k] * B[global_idy * stB + k];
    }
    C[global_idx * stC + global_idy] = accum;
}