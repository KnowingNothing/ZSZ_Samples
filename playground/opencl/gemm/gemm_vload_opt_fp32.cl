
#define BM 4
#define BN 4
#define BK 4

__kernel void gemm_vload_opt(
    __global float *A,
    __global float4 *B,
    __global float4 *C,
    int stA, int stB, int stC,
    int M, int N, int K) {
    
    int by = get_group_id(1);
    int bx = get_group_id(0);
    int ty = get_local_id(1);
    int tx = get_local_id(0);

    __local float4 ta[BM][BK];
    __local float4 tb[BK][BN];

    int ab = 4 * K * BM * by;
    int ae = ab + K;
    int bb = BN * bx;

    float4 v[4];
    for (int ii = 0; ii < 4; ++ii) {
        v[ii] = 0.0f;
    }

    const int N_float4 = N / 4;

    int i, j;
    for (i = ab, j = bb; i < ae; i += BK, j += BM * N_float4) {
        float4 temp;
        temp.x = A[0*BM*K + i + ty * K + tx];
        temp.y = A[1*BM*K + i + ty * K + tx];
        temp.z = A[2*BM*K + i + ty * K + tx];
        temp.w = A[3*BM*K + i + ty * K + tx];
        ta[ty][tx] = temp;

        tb[ty][tx] = B[j + ty * N_float4 + tx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < BK; ++k) {
            v[0] += ta[ty][k].x * tb[k][tx];
            v[1] += ta[ty][k].y * tb[k][tx];
            v[2] += ta[ty][k].z * tb[k][tx];
            v[3] += ta[ty][k].w * tb[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ii = 0; ii < 4; ++ii) {
        int tmp1 = BM * (ii + by * 4);
        int tmp2 = tmp1 + ty;
        int tmp3 = N_float4 * tmp2;
        int tmp4 = tmp3 + bx * BN;
        int tmp5 = tx;
        int tmp6 = tmp4 + tmp5;
        float4 tmp = v[ii];
        C[tmp6] = v[ii];
    }
}
