
#define BM 16
#define BN 16
#define BK 16
#define CEIL(a, b) ((a + b - 1) / b)

__kernel void gemm_memory_opt(
    __global float *A,
    __global float *B,
    __global float *C,
    int stA, int stB, int stC,
    int M, int N, int K) {
    
    int mo = get_group_id(0);
    int no = get_group_id(1);
    int mi = get_local_id(0);
    int ni = get_local_id(1);
    int MO = get_num_groups(0);
    int NO = get_num_groups(1);
    int MI = get_local_size(0);
    int NI = get_local_size(1);
    int m = mo * MI + mi;
    int n = no * NI + ni;
    
    __local float AA[BM][BK];
    __local float BB[BN][BK];

    float accum = 0;
    for (int ko = 0; ko < CEIL(K, BK); ++ko) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ki = 0; ki < BK; ki += BN) {
            int k = ko * BK + ki * BN + ni;
            AA[mi][ki * BN + ni] = k < K && m < M ? A[m * stA + k] : 0.0;
        }
        for (int ki = 0; ki < BK; ki += BM) {
            int k = ko * BK + ki * BM + mi;
            BB[ni][ki * BM + mi] = k < K && n < N ? B[n * stB + k] : 0.0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ki = 0; ki < BK; ++ki) {
            accum += AA[mi][ki] * BB[ni][ki];
        }
    }
    if (m < M && n < N)
        C[m * stC + n] = accum;
}