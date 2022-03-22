
#define BM 4
#define BN 4
#define BK 4

#define unroll_m 8
#define unroll_n 8

__kernel void gemm_multi_item_opt(
    __global float *A,
    __global float4 *B,
    __global float4 *C,
    int stA, int stB, int stC,
    int M, int N, int K) {
    
    int by = get_group_id(1);
    int bx = get_group_id(0);
    int ty = get_local_id(1);
    int tx = get_local_id(0);

#define unroll_m_float4 (unroll_m / 4)
#define unroll_n_float4 (unroll_n / 4)

    __local float4 ta[BM*unroll_m_float4][BK];
    __local float4 tb[BK*unroll_n_float4][BN];

    int ab = unroll_m * K * BM * by;
    int ae = ab + K;
    int bb = BN * bx * unroll_n_float4;

    float4 v[unroll_m][unroll_n_float4];
    for (int ii = 0; ii < unroll_m; ++ii) {
        for (int jj = 0; jj < unroll_n_float4; ++jj) {
            v[ii][jj] = 0.0f;
        }
    }

    const int N_float4 = N / 4;

    int i, j;
    for (i = ab, j = bb; i < ae; i += BK, j += BM * N_float4) {
        for (int ii = 0; ii < unroll_m_float4; ++ii) {
            float4 temp;
            temp.x = A[(4 * ii + 0) * BM * K + i + ty * K + tx];
            temp.y = A[(4 * ii + 1) * BM * K + i + ty * K + tx];
            temp.z = A[(4 * ii + 2) * BM * K + i + ty * K + tx];
            temp.w = A[(4 * ii + 3) * BM * K + i + ty * K + tx];
            ta[ii*BM + ty][tx] = temp;
        }

        for (int jj = 0; jj < unroll_n_float4; ++jj) {
            tb[jj*BN + ty][tx] = B[j + ty * N_float4 + jj * BN + tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < BK; ++k) {
            for (int ii = 0; ii < unroll_m_float4; ++ii) {
                for (int jj = 0; jj < unroll_n_float4; ++jj) {
                    float4 a = ta[ii*BM + ty][k];
                    float4 b = tb[jj*BN + k][tx];
                    v[4 * ii + 0][jj] += a.x * b;
                    v[4 * ii + 1][jj] += a.y * b;
                    v[4 * ii + 2][jj] += a.z * b;
                    v[4 * ii + 3][jj] += a.w * b;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ii = 0; ii < unroll_m; ++ii) {
        for (int jj = 0; jj < unroll_n_float4; ++jj) {
            C[N_float4 * (BM * (ii + by * unroll_m) + ty) + (bx * unroll_n_float4 + jj) * BM + tx] = v[ii][jj];
        }
    }
}

