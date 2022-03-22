
__kernel void matmul_naive(
    __global double *A,
    __global double *B,
    __global double *C,
    int stA, int stB, int stC) {
    
    int global_idx = get_global_id(0);
    int global_idy = get_global_id(1);

    float accum = 0;
    for (int k = 0; k < stA; ++k) {
        accum += A[global_idx * stA + k] * B[global_idy * stB + k];
    }
    C[global_idx * stC + global_idy] = accum;
}