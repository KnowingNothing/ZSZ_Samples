#include <immintrin.h>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <type_traits>
#include <string>
#include <chrono>

#include <mkl.h>


int M = 513;
int N = 513;
int K = 513;
bool check_result = true;
bool compare_performance = true;
int repeat = 100;

#define BLOCK_M 16
#define BLOCK_N 128
#define BLOCK_K 16

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define DATA_IN float
#define DATA_OUT float

#define CEIL(a, b) (((a) + (b)-1) / (b))


void gemm_nnn_naive(DATA_IN* A, DATA_IN* B, DATA_OUT* C, int M, int N, int K) {
    int M1 = CEIL(M, BLOCK_M);
    int N1 = CEIL(N, BLOCK_N);
    int K1 = CEIL(K, BLOCK_K);

    int M2 = CEIL(BLOCK_M, TILE_M);
    int N2 = CEIL(BLOCK_N, TILE_N);
    int K2 = CEIL(BLOCK_K, TILE_K);

    // parallel
    #pragma omp parallel for
    for (int m1n1 = 0; m1n1 < M1 * N1; ++m1n1) {
        for (int m2 = 0; m2 < M2; ++m2) {
            for (int n2 = 0; n2 < N2; ++n2) {

                DATA_IN reg_A[TILE_M * BLOCK_K];
                // relayout B
                DATA_IN reg_B[BLOCK_K * TILE_N];
                float accum[TILE_M * TILE_N] = {0.0f};

                for (int k1 = 0; k1 < K1; ++k1) {
                    
                    // load A
                    for (int m3 = 0; m3 < TILE_M; ++m3) {
                        #pragma unroll
                        for (int kk = 0; kk < BLOCK_K; ++kk) {
                            int m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                            int k = k1 * BLOCK_K + kk;
                            reg_A[m3 * BLOCK_K + kk] = m < M && k < K ? A[m * K + k] : static_cast<DATA_IN>(0.0f);
                        }
                    }
                    // load B
                    for (int kk = 0; kk < BLOCK_K; ++kk) {
                        #pragma unroll
                        for (int n3 = 0; n3 < TILE_N; ++n3) {
                            int n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3;
                            int k = k1 * BLOCK_K + kk;
                            reg_B[n3 * BLOCK_K + kk] = k < K && n < N ? B[k * N + n] : static_cast<DATA_IN>(0.0f);
                        }
                    }
                    for (int k2 = 0; k2 < K2; ++k2) {
                        // TILE_M x TIME_N x TILE_K gemm
                        for (int k3 = 0; k3 < TILE_K; ++k3) {
                            #pragma unroll
                            for (int m3 = 0; m3 < TILE_M; ++m3) {
                                #pragma unroll
                                for (int n3 = 0; n3 < TILE_N; ++n3) {
                                    accum[m3 * TILE_N + n3] = accum[m3 * TILE_N + n3] + float(reg_A[m3 * BLOCK_K + k2 * TILE_K + k3]) * float(reg_B[n3 * BLOCK_K + k2 * TILE_K + k3]);
                                }
                            }
                        }
                    }
                }
                // store C
                for (int m3 = 0; m3 < TILE_M; ++m3) {
                    for (int n3 = 0; n3 < TILE_N; ++n3) {
                        int m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                        int n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3;
                        if (m < M && n < N) {
                            C[m * N + n] = static_cast<DATA_OUT>(accum[m3 * TILE_N + n3]);
                        }
                    }
                }
            }
        }
    }
}


void gemm_nnn_vector(DATA_IN* A, DATA_IN* B, DATA_OUT* C, int M, int N, int K) {
    int M1 = CEIL(M, BLOCK_M);
    int N1 = CEIL(N, BLOCK_N);
    int K1 = CEIL(K, BLOCK_K);

    int M2 = CEIL(BLOCK_M, TILE_M);
    int N2 = CEIL(BLOCK_N, TILE_N);
    int K2 = CEIL(BLOCK_K, TILE_K);

    // parallel
    #pragma omp parallel for
    for (int m1n1 = 0; m1n1 < M1 * N1; ++m1n1) {
        for (int m2 = 0; m2 < M2; ++m2) {
            for (int n2 = 0; n2 < N2; ++n2) {
                // relayout A
                DATA_IN reg_A[TILE_M * BLOCK_K];
                DATA_IN reg_B[BLOCK_K * TILE_N];
                // relayout C
                float accum[TILE_M * TILE_N] = {0.0f};
                __m512 c[TILE_N];
                for (int n4 = 0; n4 < TILE_N; ++n4) {
                    c[n4] = _mm512_setzero_ps();
                }

                for (int k1 = 0; k1 < K1; ++k1) {
                    
                    // load A
                    for (int m3 = 0; m3 < TILE_M; ++m3) {
                        for (int kk = 0; kk < BLOCK_K; ++kk) {
                            int m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                            int k = k1 * BLOCK_K + kk;
                            reg_A[kk * TILE_M + m3] = m < M && k < K ? A[m * K + k] : static_cast<DATA_IN>(0.0f);
                        }
                    }
                    // load B
                    for (int kk = 0; kk < BLOCK_K; ++kk) {
                        for (int n3 = 0; n3 < TILE_N; ++n3) {
                            int n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3;
                            int k = k1 * BLOCK_K + kk;
                            reg_B[kk * TILE_N + n3] = k < K && n < N ? B[k * N + n] : static_cast<DATA_IN>(0.0f);
                        }
                    }
                    for (int k2 = 0; k2 < K2; ++k2) {
                        // TILE_M x TIME_N x TILE_K gemm
                        for (int n3 = 0; n3 < TILE_N; ++n3) {
                            for (int k3 = 0; k3 < TILE_K; ++k3) {
                                __m512 a, b, tmp;
                                a = _mm512_loadu_ps(reg_A + (k2 * TILE_K + k3) * TILE_M);
                                float v = reg_B[(k2 * TILE_K + k3) * TILE_N + n3];
                                b = _mm512_set1_ps(v);
                                tmp = _mm512_mul_ps(a, b);
                                c[n3] = _mm512_add_ps(tmp, c[n3]);                              
                            }
                        }
                    }
                }
                // store C
                for (int n3 = 0; n3 < TILE_N; ++n3) {
                    _mm512_storeu_ps(accum + n3 * TILE_M, c[n3]);
                    for (int m3 = 0; m3 < TILE_M; ++m3) {
                        int m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                        int n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3;
                        if (m < M && n < N) {
                            C[m * N + n] = static_cast<DATA_OUT>(accum[n3 * TILE_M + m3]);
                        }
                    }
                }
            }
        }
    }
}


void gemm_nnn_hybrid(DATA_IN* A, DATA_IN* B, DATA_OUT* C, int M, int N, int K) {
    int M1 = CEIL(M, BLOCK_M);
    int N1 = CEIL(N, BLOCK_N);
    int K1 = CEIL(K, BLOCK_K);

    int M2 = CEIL(BLOCK_M, TILE_M);
    int N2 = CEIL(BLOCK_N, TILE_N);
    int K2 = CEIL(BLOCK_K, TILE_K);

    // parallel
    #pragma omp parallel for
    for (int m1n1 = 0; m1n1 < M1 * N1; ++m1n1) {
        for (int m2 = 0; m2 < M2; ++m2) {
            for (int n2 = 0; n2 < N2; ++n2) {
                // relayout A
                DATA_IN reg_A[TILE_M * BLOCK_K];
                DATA_IN reg_B[BLOCK_K * TILE_N];
                // relayout C
                float accum1[TILE_M * 8] = {0.0f};
                float accum2[TILE_M * 8] = {0.0f};
                __m512 c[TILE_N];
                for (int n4 = 0; n4 < TILE_N; ++n4) {
                    c[n4] = _mm512_setzero_ps();
                }

                for (int k1 = 0; k1 < K1; ++k1) {
                    
                    // load A
                    for (int m3 = 0; m3 < TILE_M; ++m3) {
                        for (int kk = 0; kk < BLOCK_K; ++kk) {
                            int m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                            int k = k1 * BLOCK_K + kk;
                            reg_A[kk * TILE_M + m3] = m < M && k < K ? A[m * K + k] : static_cast<DATA_IN>(0.0f);
                        }
                    }
                    // load B
                    for (int kk = 0; kk < BLOCK_K; ++kk) {
                        for (int n3 = 0; n3 < TILE_N; ++n3) {
                            int n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3;
                            int k = k1 * BLOCK_K + kk;
                            reg_B[kk * TILE_N + n3] = k < K && n < N ? B[k * N + n] : static_cast<DATA_IN>(0.0f);
                        }
                    }

                    for (int k2 = 0; k2 < K2; ++k2) {
                        // TILE_M x TIME_N x TILE_K gemm
                        for (int k3 = 0; k3 < TILE_K; ++k3) {
                            for (int n3 = 0; n3 < 8; ++n3) {
                                __m512 a, b, tmp;
                                a = _mm512_loadu_ps(reg_A + (k2 * TILE_K + k3) * TILE_M);
                                float v = reg_B[(k2 * TILE_K + k3) * TILE_N + n3];
                                b = _mm512_set1_ps(v);
                                tmp = _mm512_mul_ps(a, b);
                                for (int m3 = 0; m3 < TILE_M; ++m3) {
                                    accum2[n3 * TILE_M + m3] = accum2[n3 * TILE_M + m3] + float(reg_A[(k2 * TILE_K + k3) * TILE_M + m3]) * float(reg_B[(k2 * TILE_K + k3) * TILE_N + n3 + 8]);
                                }
                                c[n3] = _mm512_add_ps(tmp, c[n3]);                              
                            }
                        }
                    }

                    
                }
                // store C
                for (int n3 = 0; n3 < 8; ++n3) {
                    _mm512_storeu_ps(accum1 + n3 * TILE_M, c[n3]);
                    for (int m3 = 0; m3 < TILE_M; ++m3) {
                        int m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                        int n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3 + 8;
                        if (m < M && n < N) {
                            C[m * N + n] = static_cast<DATA_OUT>(accum2[n3 * TILE_M + m3]);
                        }
                        m = (m1n1 / N1) * BLOCK_M + m2 * TILE_M + m3;
                        n = (m1n1 % N1) * BLOCK_N + n2 * TILE_N + n3;
                        if (m < M && n < N) {
                            C[m * N + n] = static_cast<DATA_OUT>(accum1[n3 * TILE_M + m3]);
                        }
                    }
                }                
            }
        }
    }
}


std::string usage()
{
    return "Usage:\ngemm_hybrid_intel M N K check_result(0 or 1) compare_performance(0 or 1)\n";
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
            M = std::atoi(argv[1]);
            N = std::atoi(argv[2]);
            K = std::atoi(argv[3]);
        }
        else if (argc == 5)
        {
            M = std::atoi(argv[1]);
            N = std::atoi(argv[2]);
            K = std::atoi(argv[3]);
            check_result = bool(std::atoi(argv[4]));
        }
        else if (argc == 6)
        {
            M = std::atoi(argv[1]);
            N = std::atoi(argv[2]);
            K = std::atoi(argv[3]);
            check_result = bool(std::atoi(argv[4]));
            compare_performance = bool(std::atoi(argv[5]));
        }
    }

    std::cout << "M=" << M << ",N=" << N << ",K=" << K << "\n"
              << std::flush;

    DATA_IN *A, *B;
    DATA_OUT *C, *golden_C;

    A = (DATA_IN *)malloc(M * K * sizeof(DATA_IN));
    B = (DATA_IN *)malloc(K * N * sizeof(DATA_IN));
    C = (DATA_OUT *)malloc(M * N * sizeof(DATA_OUT));
    golden_C = (DATA_OUT *)malloc(M * N * sizeof(DATA_OUT));

    std::cout << "Intializing data..." << std::flush;
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            A[i * K + k] = static_cast<DATA_IN>(2.0); //static_cast<DATA_IN>(((i * k / 3.0) + rand() % 999) / 77.0);
        }
    }
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            B[k * N + i] = static_cast<DATA_IN>(3.0); //static_cast<DATA_IN>(((i * k / 3.0) + rand() % 999) / 77.0);
        }
    }
    std::cout << "  Done!\n"
              << std::flush;

    std::cout << "Block size: (" << BLOCK_M << ", " << BLOCK_N << ", " << BLOCK_K << ")\n"
              << std::flush;
    std::cout << "Tile size: (" << TILE_M << ", " << TILE_N << ", " << TILE_K << ")\n"
              << std::flush;
    std::cout << "Calling kernel..." << std::flush;
    gemm_nnn_hybrid(A, B, C, M, N, K);

    std::cout << "  Done!\n"
              << std::flush;

    if (check_result)
    {
        std::cout << "Calculating golden results..." << std::flush;
        for (int io = 0; io < CEIL(M, 16); ++io)
        {
            for (int jo = 0; jo < CEIL(N, 16); ++jo)
            {
                float accum[16][16] = {{0.0f}};
                for (int ko = 0; ko < CEIL(K, 16); ++ko)
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
                                if (i < M && j < N && k < K)
                                {
                                    accum[ii][ji] += float(A[i * K + k]) * float(B[k * N + j]);
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
                        if (i < M && j < N)
                        {
                            golden_C[i * N + j] = accum[ii][ji];
                        }
                    }
                }
            }
        }
        std::cout << "  Done!\n"
                  << std::flush;

        std::cout << "Check results connectness..." << std::flush;
        int errors = 0;
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                if (std::abs(float(C[i * N + j]) - float(golden_C[i * N + j])) >= 1e-5)
                {
                    errors += 1;
                    if (errors <= 9)
                    {
                        std::cout << "\nres=" << float(C[i * N + j]) << ", golden=" << float(golden_C[i * N + j]) << std::flush;
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
            if (M * N <= 300)
            {
                for (int i = 0; i < M; ++i)
                {
                    for (int j = 0; j < N; ++j)
                    {
                        std::cout << "(" << float(C[i * N + j]) << "," << float(golden_C[i * N + j]) << ") " << std::flush;
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
        float mkl_time, naive_time, vector_time, hybrid_time;
        {
            float alpha = 1.0f;
            float beta = 0.0f;
            // warm up
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
            std::chrono::high_resolution_clock::time_point t1;
            t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; ++i) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            mkl_time = float(time_span.count() / repeat * 1e3);
        }

        {
            // warm up
            gemm_nnn_naive(A, B, C, M, N, K);
            std::chrono::high_resolution_clock::time_point t1;
            t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; ++i) {
                gemm_nnn_naive(A, B, C, M, N, K);
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            naive_time = float(time_span.count() / repeat * 1e3);
        }

        {
            // warm up
            gemm_nnn_vector(A, B, C, M, N, K);
            std::chrono::high_resolution_clock::time_point t1;
            t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; ++i) {
                gemm_nnn_vector(A, B, C, M, N, K);
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            vector_time = float(time_span.count() / repeat * 1e3);
        }

        {
            // warm up
            gemm_nnn_hybrid(A, B, C, M, N, K);
            std::chrono::high_resolution_clock::time_point t1;
            t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < repeat; ++i) {
                gemm_nnn_hybrid(A, B, C, M, N, K);
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            hybrid_time = float(time_span.count() / repeat * 1e3);
        }
        std::cout << "MKL: " << mkl_time << "ms, Naive: " << naive_time << "ms, Vector: " << vector_time << "ms, Hybrid: " << hybrid_time << "ms, Relative Perf: " << mkl_time / hybrid_time << "\n"
                  << std::flush;
    }

    free(A);
    free(B);
    free(C);
    free(golden_C);
    return 0;
}