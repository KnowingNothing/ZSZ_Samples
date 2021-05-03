#define BM 4
#define BN 4
#define BK 4

#define unroll_m 8
#define unroll_n 8

#define PRO_LOAD(istart, jstart)                                               \
  {                                                                            \
    for (int ii = 0; ii < unroll_m_float4; ++ii) {                             \
      float4 *temp = tempA + ii;                                               \
      temp->x = A[(4 * ii + 0) * BM * K + istart + ty * K + tx];               \
      temp->y = A[(4 * ii + 1) * BM * K + istart + ty * K + tx];               \
      temp->z = A[(4 * ii + 2) * BM * K + istart + ty * K + tx];               \
      temp->w = A[(4 * ii + 3) * BM * K + istart + ty * K + tx];               \
    }                                                                          \
                                                                               \
    for (int jj = 0; jj < unroll_n_float4; ++jj) {                             \
      tempB[jj] = B[jstart + ty * N_float4 + jj * BN + tx];                    \
    }                                                                          \
  }

#define STORE_TO_SMEM                                                          \
  {                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
    for (int ii = 0; ii < unroll_m_float4; ++ii) {                             \
      ta[ii * BM + ty][tx] = tempA[ii];                                        \
    }                                                                          \
    for (int jj = 0; jj < unroll_n_float4; ++jj) {                             \
      tb[jj * BN + ty][tx] = tempB[jj];                                        \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
  }

#define COMPUTE                                                                \
  {                                                                            \
    for (int k = 0; k < BK; ++k) {                                             \
      for (int ii = 0; ii < unroll_m_float4; ++ii) {                           \
        for (int jj = 0; jj < unroll_n_float4; ++jj) {                         \
          float4 a = ta[ii * BM + ty][k];                                      \
          float4 b = tb[jj * BN + k][tx];                                      \
          v[4 * ii + 0][jj] += a.x * b;                                        \
          v[4 * ii + 1][jj] += a.y * b;                                        \
          v[4 * ii + 2][jj] += a.z * b;                                        \
          v[4 * ii + 3][jj] += a.w * b;                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

__kernel void gemm_opt(__global float *A, __global float4 *B,
                       __global float4 *C, int stA, int stB, int stC, int M,
                       int N, int K) {
  int by = get_group_id(1);
  int bx = get_group_id(0);
  int ty = get_local_id(1);
  int tx = get_local_id(0);

#define unroll_m_float4 (unroll_m / 4)
#define unroll_n_float4 (unroll_n / 4)

  __local float4 ta[BM * unroll_m_float4][BK];
  __local float4 tb[BK * unroll_n_float4][BN];

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
  float4 tempA[unroll_m_float4];
  float4 tempB[unroll_n_float4];

  PRO_LOAD(ab, bb);

  int i, j;
#pragma unroll 1
  for (i = ab, j = bb; i < ae - BK; i += BK, j += BK * N_float4) {
    STORE_TO_SMEM
    PRO_LOAD(i + BK, j + BK * N_float4)
    COMPUTE
  }

  STORE_TO_SMEM
  COMPUTE

#pragma unroll 1
  for (int ii = 0; ii < unroll_m; ++ii) {
    for (int jj = 0; jj < unroll_n_float4; ++jj) {
      C[N_float4 * (BM * (ii + by * unroll_m) + ty) +
        (bx * unroll_n_float4 + jj) * BM + tx] = v[ii][jj];
    }
  }
}