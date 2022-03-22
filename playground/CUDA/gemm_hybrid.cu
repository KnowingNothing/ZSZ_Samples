#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <type_traits>
#include <string>

#include <cublas_v2.h>

#define WARP_SIZE 32

int host_M = 512;
int host_N = 512;
int host_K = 512;
bool check_result = true;
bool compare_performance = false;
int repeat = 100;

#define BLOCK_M 64
#define BLOCK_N 32
#define BLOCK_K 32

#define WARP_M 16
#define WARP_N 16
#define WARP_K 16

#define DATA_IN __half
#define DATA_OUT __half

#define CEIL(a, b) (((a) + (b)-1) / (b))

const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

#define CUBLASCHECK(expression)                                     \
    {                                                               \
        cublasStatus_t status = (expression);                       \
        if (status != CUBLAS_STATUS_SUCCESS)                        \
        {                                                           \
            std::cerr << "Error on line " << __LINE__ << ": "       \
                      << cublasGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    }

#define CUDACHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess)                                             \
        {                                                                 \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

template <int length>
__device__ void load_matrix_global_n(
    DATA_IN *src, DATA_IN dst[length],
    int offset, int ldm, int height, int width, int src_len)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
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
__global__ void gemm_hybrid_nnn(DATA_IN *A, DATA_IN *B, DATA_OUT *C, int M, int N, int K)
{
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
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, DATA_OUT> accum_frag;
    nvcuda::wmma::fill_fragment(accum_frag, static_cast<DATA_OUT>(0.0f));

#define COND (warp_id % 16 / 4 == 0)

    for (int k1 = 0; k1 < K1; ++k1)
    {
        int block_offset_A = block_m * BLOCK_M * K + k1 * BLOCK_K;
        int block_offset_B = k1 * BLOCK_K * N + block_n * BLOCK_N;
        for (int k2 = 0; k2 < K2; ++k2)
        {
            int warp_offset_A = warp_m * WARP_M * K + k2 * WARP_K;
            int warp_offset_B = k2 * WARP_K * N + warp_n * WARP_N;
            if (COND)
            {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, DATA_IN, nvcuda::wmma::row_major> A_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, DATA_IN, nvcuda::wmma::row_major> B_frag;
                nvcuda::wmma::load_matrix_sync(A_frag, A + block_offset_A + warp_offset_A, K);
                nvcuda::wmma::load_matrix_sync(B_frag, B + block_offset_B + warp_offset_B, N);
                nvcuda::wmma::mma_sync(accum_frag, A_frag, B_frag, accum_frag);
            }
            else
            {
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
                for (int m4 = 0; m4 < M4; ++m4)
                {
                    for (int n4 = 0; n4 < N4; ++n4)
                    {
                        // omit for loop k4 = 1
                        for (int k4 = 0; k4 < K4; ++k4)
                        {
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

    if (COND)
    {
        int block_offset = block_m * BLOCK_M * N + block_n * BLOCK_N;
        int warp_offset = warp_m * WARP_M * N + warp_n * WARP_N;
        nvcuda::wmma::store_matrix_sync(C + block_offset + warp_offset, accum_frag, N, nvcuda::wmma::mem_row_major);
    }
    else
    {
#define M3 8
#define N3 4
        int thread_id_within_warp = tx % WARP_SIZE;
        int thread_m = thread_id_within_warp / N3;
        int thread_n = thread_id_within_warp % N3;
        int M4 = CEIL(WARP_M, M3);
        int N4 = CEIL(WARP_N, N3);
        for (int m4 = 0; m4 < M4; ++m4)
        {
            for (int n4 = 0; n4 < N4; ++n4)
            {
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

std::string usage()
{
    return "Usage:\ngemm_hybrid M N K check_result(0 or 1) compare_performance(0 or 1)\n";
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        if (argc != 6 && argc != 5 && argc != 4)
        {
            std::cout << "Error arguments:\n"
                      << std::flush;
            std::cout << usage() << std::flush;
            return 1;
        }
        else if (argc == 4)
        {
            host_M = std::atoi(argv[1]);
            host_N = std::atoi(argv[2]);
            host_K = std::atoi(argv[3]);
        }
        else if (argc == 5)
        {
            host_M = std::atoi(argv[1]);
            host_N = std::atoi(argv[2]);
            host_K = std::atoi(argv[3]);
            check_result = bool(std::atoi(argv[4]));
        }
        else if (argc == 6)
        {
            host_M = std::atoi(argv[1]);
            host_N = std::atoi(argv[2]);
            host_K = std::atoi(argv[3]);
            check_result = bool(std::atoi(argv[4]));
            compare_performance = bool(std::atoi(argv[5]));
        }
    }

    std::cout << "M=" << host_M << ",N=" << host_N << ",K=" << host_K << "\n"
              << std::flush;

    static_assert(WARP_M * WARP_N % WARP_SIZE == 0, "Warp level size should be divisible by warp size.\n");
    static_assert((WARP_M == 16) && (WARP_N == 16) && (WARP_K == 16), "Warp level size should 16x16x16.\n");
    DATA_IN *host_A, *host_B, *dev_A, *dev_B;
    DATA_OUT *host_C, *dev_C, *golden_C;
    CUDACHECK(cudaMalloc((void **)&dev_A, host_M * host_K * sizeof(DATA_IN)));
    CUDACHECK(cudaMalloc((void **)&dev_B, host_K * host_N * sizeof(DATA_IN)));
    CUDACHECK(cudaMalloc((void **)&dev_C, host_M * host_N * sizeof(DATA_IN)));

    host_A = (DATA_IN *)malloc(host_M * host_K * sizeof(DATA_IN));
    host_B = (DATA_IN *)malloc(host_K * host_N * sizeof(DATA_IN));
    host_C = (DATA_OUT *)malloc(host_M * host_N * sizeof(DATA_OUT));
    golden_C = (DATA_OUT *)malloc(host_M * host_N * sizeof(DATA_OUT));

    std::cout << "Intializing data on host..." << std::flush;
    for (int i = 0; i < host_M; ++i)
    {
        for (int k = 0; k < host_K; ++k)
        {
            host_A[i * host_K + k] = static_cast<DATA_IN>(1.0); //static_cast<DATA_IN>(((i * k / 3.0) + rand() % 999) / 77.0);
        }
    }
    for (int i = 0; i < host_N; ++i)
    {
        for (int k = 0; k < host_K; ++k)
        {
            host_B[k * host_N + i] = static_cast<DATA_IN>(2.0); //static_cast<DATA_IN>(((i * k / 3.0) + rand() % 999) / 77.0);
        }
    }
    std::cout << "  Done!\n"
              << std::flush;

    std::cout << "Copying data to device..." << std::flush;
    CUDACHECK(cudaMemcpy(dev_A, host_A, host_M * host_K * sizeof(DATA_IN), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(dev_B, host_B, host_K * host_N * sizeof(DATA_IN), cudaMemcpyHostToDevice));
    std::cout << "  Done!\n";

    dim3 grid(CEIL(host_M, BLOCK_M), CEIL(host_N, BLOCK_N));
    dim3 block(CEIL(BLOCK_M, WARP_M) * CEIL(BLOCK_N, WARP_N) * WARP_SIZE);
    std::cout << "Grid size: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n"
              << std::flush;
    std::cout << "Block size: (" << block.x << ", " << block.y << ", " << block.z << ")\n"
              << std::flush;
    std::cout << "Calling kernel..." << std::flush;
    gemm_hybrid_nnn<<<grid, block>>>(dev_A, dev_B, dev_C, host_M, host_N, host_K);
    CUDACHECK(cudaGetLastError());
    std::cout << "  Done!\n"
              << std::flush;

    std::cout << "Copying results to host..." << std::flush;
    CUDACHECK(cudaMemcpy(host_C, dev_C, host_M * host_N * sizeof(DATA_OUT), cudaMemcpyDeviceToHost));
    std::cout << "  Done!\n";

    if (check_result)
    {
        std::cout << "Calculating golden results..." << std::flush;
        for (int io = 0; io < CEIL(host_M, 16); ++io)
        {
            for (int jo = 0; jo < CEIL(host_N, 16); ++jo)
            {
                float accum[16][16] = {{0.0f}};
                for (int ko = 0; ko < CEIL(host_K, 16); ++ko)
                {
                    for (int ii = 0; ii < 16; ++ii)
                    {
                        for (int ji = 0; ji < 16; ++ji)
                        {
                            for (int ki = 0; ki < 16; ++ki)
                            {
                                int i = io * 16 + ii;
                                int j = jo * 16 + ji;
                                int k = ko * 16 + ki;
                                if (i < host_M && j < host_N && k < host_K)
                                {
                                    accum[ii][ji] += float(host_A[i * host_K + k]) * float(host_B[k * host_N + j]);
                                }
                            }
                        }
                    }
                }
                for (int ii = 0; ii < 16; ++ii)
                {
                    for (int ji = 0; ji < 16; ++ji)
                    {
                        int i = io * 16 + ii;
                        int j = jo * 16 + ji;
                        if (i < host_M && j < host_N)
                        {
                            golden_C[i * host_N + j] = accum[ii][ji];
                        }
                    }
                }
            }
        }
        std::cout << "  Done!\n"
                  << std::flush;

        std::cout << "Check results connectness..." << std::flush;
        int errors = 0;
        for (int i = 0; i < host_M; ++i)
        {
            for (int j = 0; j < host_N; ++j)
            {
                if (std::abs(float(host_C[i * host_N + j]) - float(golden_C[i * host_N + j])) >= 1e-5)
                {
                    errors += 1;
                    if (errors <= 9)
                    {
                        std::cout << "\nres=" << float(host_C[i * host_N + j]) << ", golden=" << float(golden_C[i * host_N + j]) << std::flush;
                    }
                    else if (errors == 10)
                    {
                        std::cout << ".\n.\n.\n"
                                  << std::flush;
                    }
                }
            }
        }
        if (errors)
        {
            std::cout << errors << " errors!\n"
                      << std::flush;
            if (host_M * host_N <= 300)
            {
                for (int i = 0; i < host_M; ++i)
                {
                    for (int j = 0; j < host_N; ++j)
                    {
                        std::cout << "(" << float(host_C[i * host_N + j]) << "," << float(golden_C[i * host_N + j]) << ") " << std::flush;
                    }
                    std::cout << "\n"
                              << std::flush;
                }
            }
        }
        else
        {
            std::cout << "  Pass!\n"
                      << std::flush;
        }
    }

    if (compare_performance)
    {
        float cublas_time, hybrid_time;
        {
            cudaEvent_t start, stop;
            float elapsed_time;
            CUDACHECK(cudaEventCreate(&start));
            CUDACHECK(cudaEventCreate(&stop));
            cublasHandle_t cublas_h;
            CUBLASCHECK(cublasCreate(&cublas_h));

            // call cublas
            DATA_IN alpha = static_cast<DATA_IN>(1.0f);
            DATA_IN beta = static_cast<DATA_IN>(0.0f);
            // first warm up
            CUBLASCHECK(cublasHgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, host_N, host_M, host_K, &alpha, dev_B, host_N, dev_A, host_K, &beta, dev_C, host_N));
            CUDACHECK(cudaEventRecord(start));
            for (int i = 0; i < repeat; ++i)
            {
                CUBLASCHECK(cublasHgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, host_N, host_M, host_K, &alpha, dev_B, host_N, dev_A, host_K, &beta, dev_C, host_N));
            }
            CUDACHECK(cudaEventRecord(stop));
            CUDACHECK(cudaEventSynchronize(stop));
            CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); // ms
            cublas_time = elapsed_time / repeat;
            CUDACHECK(cudaEventDestroy(start));
            CUDACHECK(cudaEventDestroy(stop));
        }
        {
            cudaEvent_t start, stop;
            float elapsed_time;
            CUDACHECK(cudaEventCreate(&start));
            CUDACHECK(cudaEventCreate(&stop));

            // first warm up
            gemm_hybrid_nnn<<<grid, block>>>(dev_A, dev_B, dev_C, host_M, host_N, host_K);
            CUDACHECK(cudaEventRecord(start));
            for (int i = 0; i < repeat; ++i)
            {
                gemm_hybrid_nnn<<<grid, block>>>(dev_A, dev_B, dev_C, host_M, host_N, host_K);
            }
            CUDACHECK(cudaEventRecord(stop));
            CUDACHECK(cudaEventSynchronize(stop));
            CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); // ms
            hybrid_time = elapsed_time / repeat;
            CUDACHECK(cudaEventDestroy(start));
            CUDACHECK(cudaEventDestroy(stop));
        }
        std::cout << "CuBlas: " << cublas_time << "ms, Hybrid: " << hybrid_time << "ms, Relative Perf: " << cublas_time / hybrid_time << "\n"
                  << std::flush;
    }

    free(host_A);
    free(host_B);
    free(host_C);
    free(golden_C);
    CUDACHECK(cudaFree(dev_A));
    CUDACHECK(cudaFree(dev_B));
    CUDACHECK(cudaFree(dev_C));
    return 0;
}