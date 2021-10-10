#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <type_traits>

#define WARP_SIZE 32


#define M 512
#define N 512
#define K 512

#define BLOCK_M 64
#define BLOCK_N 32
#define BLOCK_K 32

#define WARP_M 16
#define WARP_N 16
#define WARP_K 16

#define DATA_IN __half
#define DATA_OUT __half
#define CHECK_RESULT


#define CEIL(a, b) (((a) + (b) - 1) / (b))


#define CUDACHECK(cmd)                                                    \
    do {                                                                  \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess) {                                           \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)


template<int length>
__device__ void load_matrix_global_n(
    DATA_IN* src, DATA_IN dst[length],
    int offset, int ldm, int height, int width, int src_len) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            dst[i * width + j] = offset + i * ldm + j < src_len ? src[offset + i * ldm + j] : static_cast<DATA_IN>(0.0f);
        }
    }
}


// gemm
// A: row major MxK
// B: row major KxN
// C: row major MxN
// block x dimension for M
// block y dimension for N
// thread x dimension for MxN
__global__ void gemm_hybrid_nnn(DATA_IN* A, DATA_IN* B, DATA_OUT* C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int total_warps = CEIL(BLOCK_M, WARP_M) * CEIL(BLOCK_N, WARP_N);

    int block_m = bx;
    int block_n = by;
    int warp_m = warp_id / CEIL(BLOCK_N, WARP_N);
    int warp_n = warp_id % CEIL(BLOCK_N, WARP_N);

    int K1 = CEIL(K, BLOCK_K);
    int K2 = CEIL(BLOCK_K, WARP_K);

    float accum[WARP_M * WARP_N / WARP_SIZE] = {0.0f};

    #define COND (warp_id * 2 < total_warps)

    for (int k1 = 0; k1 < K1; ++k1) {
        int block_offset_A = block_m * BLOCK_M * K + k1 * BLOCK_K;
        int block_offset_B = k1 * BLOCK_K * N + block_n * BLOCK_N;
        for (int k2 = 0; k2 < K2; ++k2) {
            int warp_offset_A = warp_m * WARP_M * K + k2 * WARP_K;
            int warp_offset_B = k2 * WARP_K * N + warp_n * WARP_N;
            if (COND) {
                // fma
                // warp orgnization: 2x16
                #define M3 2
                #define N3 16
                #define K3 1
                int M4 = CEIL(WARP_M, M3);
                int N4 = CEIL(WARP_N, N3);
                int K4 = WARP_K;
                int thread_id_within_warp = tx % WARP_SIZE;
                int thread_m = thread_id_within_warp / N3;
                int thread_n = thread_id_within_warp % N3;
                int thread_offset_A = thread_m * M4 * K;
                int thread_offset_B = thread_n * N4;
                // load A
                DATA_IN reg_A[CEIL(WARP_M, M3) * CEIL(WARP_K, K3)];
                load_matrix_global_n<CEIL(WARP_M, M3) * CEIL(WARP_K, K3)>(
                    A, reg_A, block_offset_A + warp_offset_A + thread_offset_A,
                    K, M4, K4, M * K);
                // load B
                DATA_IN reg_B[CEIL(WARP_K, K3) * CEIL(WARP_N, N3)];
                load_matrix_global_n<CEIL(WARP_K, K3) * CEIL(WARP_N, N3)>(
                    B, reg_B, block_offset_B + warp_offset_B + thread_offset_B,
                    N, K4, N4, K * N);
                for (int m4 = 0; m4 < M4; ++m4) {
                    for (int n4 = 0; n4 < N4; ++n4) {
                        // omit for loop k4 = 1
                        for (int k4 = 0; k4 < K4; ++k4) {
                            accum[m4 * N4 + n4] = accum[m4 * N4 + n4] + float(reg_A[m4 * K4 + k4]) * float(reg_B[k4 * N4 + n4]);
                        }
                    }
                }
                #undef M3
                #undef N3
                #undef K3
            } else {
                // fma
                // warp orgnization: 8x4
                #define M3 8
                #define N3 4
                #define K3 1
                int M4 = CEIL(WARP_M, M3);
                int N4 = CEIL(WARP_N, N3);
                int K4 = WARP_K;
                int thread_id_within_warp = tx % WARP_SIZE;
                int thread_m = thread_id_within_warp / N3;
                int thread_n = thread_id_within_warp % N3;
                int thread_offset_A = thread_m * M4 * K;
                int thread_offset_B = thread_n * N4;
                // load A
                DATA_IN reg_A[CEIL(WARP_M, M3) * CEIL(WARP_K, K3)];
                load_matrix_global_n<CEIL(WARP_M, M3) * CEIL(WARP_K, K3)>(
                    A, reg_A, block_offset_A + warp_offset_A + thread_offset_A,
                    K, M4, K4, M * K);
                // load B
                DATA_IN reg_B[CEIL(WARP_K, K3) * CEIL(WARP_N, N3)];
                load_matrix_global_n<CEIL(WARP_K, K3) * CEIL(WARP_N, N3)>(
                    B, reg_B, block_offset_B + warp_offset_B + thread_offset_B,
                    N, K4, N4, K * N);
                for (int m4 = 0; m4 < M4; ++m4) {
                    for (int n4 = 0; n4 < N4; ++n4) {
                        // omit for loop k4 = 1
                        for (int k4 = 0; k4 < K4; ++k4) {
                            accum[m4 * N4 + n4] = accum[m4 * N4 + n4] + float(reg_A[m4 * K4 + k4]) * float(reg_B[k4 * N4 + n4]);
                        }
                    }
                }
                #undef M3
                #undef N3
                #undef K3
            }
        }
    }

    if (COND) {
        #define M3 2
        #define N3 16
        int thread_id_within_warp = tx % WARP_SIZE;
        int thread_m = thread_id_within_warp / N3;
        int thread_n = thread_id_within_warp % N3;
        int M4 = CEIL(WARP_M, M3);
        int N4 = CEIL(WARP_N, N3);
        for (int m4 = 0; m4 < M4; ++m4) {
            for (int n4 = 0; n4 < N4; ++n4) {
                int idx_m = (block_m * BLOCK_M + warp_m * WARP_M + thread_m * M4 + m4);
                int idx_n = (block_n * BLOCK_N + warp_n * WARP_N + thread_n * N4 + n4);
                C[idx_m * N + idx_n] = static_cast<DATA_OUT>(accum[m4 * N4 + n4]);
            }
        }
        #undef M3
        #undef N3
    } else {
        #define M3 8
        #define N3 4
        int thread_id_within_warp = tx % WARP_SIZE;
        int thread_m = thread_id_within_warp / N3;
        int thread_n = thread_id_within_warp % N3;
        int M4 = CEIL(WARP_M, M3);
        int N4 = CEIL(WARP_N, N3);
        for (int m4 = 0; m4 < M4; ++m4) {
            for (int n4 = 0; n4 < N4; ++n4) {
                int idx_m = (block_m * BLOCK_M + warp_m * WARP_M + thread_m * M4 + m4);
                int idx_n = (block_n * BLOCK_N + warp_n * WARP_N + thread_n * N4 + n4);
                C[idx_m * N + idx_n] = static_cast<DATA_OUT>(accum[m4 * N4 + n4]);
            }
        }
        #undef M3
        #undef N3
    }

    #undef COND
}


int main(int argc, char* argv[]) {
    static_assert(WARP_M * WARP_N % WARP_SIZE == 0, "Warp level size should be divisible by warp size.\n");
    DATA_IN* host_A, *host_B, *dev_A, *dev_B;
    DATA_OUT* host_C, *dev_C, *golden_C;
    CUDACHECK(cudaMalloc((void**)&dev_A, M*K*sizeof(DATA_IN)));
    CUDACHECK(cudaMalloc((void**)&dev_B, K*N*sizeof(DATA_IN)));
    CUDACHECK(cudaMalloc((void**)&dev_C, M*N*sizeof(DATA_IN)));

    host_A = (DATA_IN*)malloc(M*K*sizeof(DATA_IN));
    host_B = (DATA_IN*)malloc(K*N*sizeof(DATA_IN));
    host_C = (DATA_OUT*)malloc(M*N*sizeof(DATA_OUT));
    golden_C = (DATA_OUT*)malloc(M*N*sizeof(DATA_OUT));

    std::cout << "Intializing data on host..." << std::flush;
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            host_A[i * K + k] = static_cast<DATA_IN>(1.0); //static_cast<DATA_IN>(((i * k / 3.0) + rand() % 999) / 77.0);
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            host_B[k * N + i] = static_cast<DATA_IN>(2.0); //static_cast<DATA_IN>(((i * k / 3.0) + rand() % 999) / 77.0);
        }
    }
    std::cout << "  Done!\n" << std::flush;

    std::cout << "Copying data to device..." << std::flush;
    CUDACHECK(cudaMemcpy(dev_A, host_A, M*K*sizeof(DATA_IN), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dev_B, host_B, K*N*sizeof(DATA_IN), cudaMemcpyHostToDevice));
    std::cout << "  Done!\n";

    dim3 grid(CEIL(M, BLOCK_M), CEIL(N, BLOCK_N));
    dim3 block(CEIL(BLOCK_M, WARP_M) * CEIL(BLOCK_N, WARP_N) * WARP_SIZE);
    std::cout << "Grid size: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n" << std::flush;
    std::cout << "Block size: (" << block.x << ", " << block.y << ", " << block.z << ")\n" << std::flush;
    std::cout << "Calling kernel..." << std::flush;
    gemm_hybrid_nnn<<<grid, block>>>(dev_A, dev_B, dev_C);
    CUDACHECK(cudaGetLastError());
    std::cout << "  Done!\n" << std::flush;

    std::cout << "Copying results to host..." << std::flush;
    CUDACHECK(cudaMemcpy(host_C, dev_C, M*N*sizeof(DATA_OUT), cudaMemcpyDeviceToHost));
    std::cout << "  Done!\n";

#ifdef CHECK_RESULT
    std::cout << "Calculating golden results..." << std::flush;
    for (int io = 0; io < CEIL(M, 16); ++io) {
        for (int jo = 0; jo < CEIL(N, 16); ++jo) {
            float accum[16][16] = {{0.0f}};
            for (int ko = 0; ko < CEIL(K, 16); ++ko) {
                for (int ii = 0; ii < 16; ++ii) {
                    for (int ji = 0; ji < 16; ++ji) {
                        for (int ki = 0; ki < 16; ++ki) {
                            int i = io * 16 + ii;
                            int j = jo * 16 + ji;
                            int k = ko * 16 + ki;
                            if (i < M && j < N && k < K) {
                                accum[ii][ji] += float(host_A[i * K + k]) * float(host_B[k * N + j]);
                            }
                        }
                    }
                }
            }
            for (int ii = 0; ii < 16; ++ii) {
                for (int ji = 0; ji < 16; ++ji) {
                    int i = io * 16 + ii;
                    int j = jo * 16 + ji;
                    if (i < M && j < N) {
                        golden_C[i * N + j] = accum[ii][ji];
                    }
                }
            }
        }
    }
    std::cout << "  Done!\n" << std::flush;

    std::cout << "Check results connectness..." << std::flush;
    int errors = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(float(host_C[i * N + j]) - float(golden_C[i * N + j])) >= 1e-5) {
                errors += 1;
                if (errors <= 9) {
                    std::cout << "\nres=" << float(host_C[i * N + j]) << ", golden=" << float(golden_C[i * N + j]) << std::flush;
                } else if (errors == 10) {
                    std::cout << ".\n.\n.\n" << std::flush;
                }
            }
        }
    }
    if (errors) {
        std::cout << errors << " errors!\n" << std::flush;
        if (M * N <= 300) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    std::cout << "(" << float(host_C[i * N + j]) << "," << float(golden_C[i * N + j]) << ") " << std::flush;
                }
                std::cout << "\n" << std::flush;
            }
        }
    } else {
        std::cout << "  Pass!\n" << std::flush;
    }
#endif
    
    return 0;
}