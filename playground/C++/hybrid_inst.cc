#include <immintrin.h>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <type_traits>
#include <string>
#include <chrono>


int repeat = 100;


void fma_kernel() {
    #pragma omp parallel for
    for (int i = 0; i < 10240; ++i) {
        float A[80] = {0.0f};
        float B[80] = {1.0f};
        float C[80] = {0.0f};
        for (int j = 0; j < 1024; ++j) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                C[x * 16 + 0] = C[x * 16 + 0] + A[x * 16 + 0] * B[x * 16 + 0];
                C[x * 16 + 1] = C[x * 16 + 1] + A[x * 16 + 1] * B[x * 16 + 1];
                C[x * 16 + 2] = C[x * 16 + 2] + A[x * 16 + 2] * B[x * 16 + 2];
                C[x * 16 + 3] = C[x * 16 + 3] + A[x * 16 + 3] * B[x * 16 + 3];
                C[x * 16 + 4] = C[x * 16 + 4] + A[x * 16 + 4] * B[x * 16 + 4];
                C[x * 16 + 5] = C[x * 16 + 5] + A[x * 16 + 5] * B[x * 16 + 5];
                C[x * 16 + 6] = C[x * 16 + 6] + A[x * 16 + 6] * B[x * 16 + 6];
                C[x * 16 + 7] = C[x * 16 + 7] + A[x * 16 + 7] * B[x * 16 + 7];

                C[x * 16 + 0 + 8] = C[x * 16 + 0 + 8] + A[x * 16 + 0 + 8] * B[x * 16 + 0 + 8];
                C[x * 16 + 1 + 8] = C[x * 16 + 1 + 8] + A[x * 16 + 1 + 8] * B[x * 16 + 1 + 8];
                C[x * 16 + 2 + 8] = C[x * 16 + 2 + 8] + A[x * 16 + 2 + 8] * B[x * 16 + 2 + 8];
                C[x * 16 + 3 + 8] = C[x * 16 + 3 + 8] + A[x * 16 + 3 + 8] * B[x * 16 + 3 + 8];
                C[x * 16 + 4 + 8] = C[x * 16 + 4 + 8] + A[x * 16 + 4 + 8] * B[x * 16 + 4 + 8];
                C[x * 16 + 5 + 8] = C[x * 16 + 5 + 8] + A[x * 16 + 5 + 8] * B[x * 16 + 5 + 8];
                C[x * 16 + 6 + 8] = C[x * 16 + 6 + 8] + A[x * 16 + 6 + 8] * B[x * 16 + 6 + 8];
                C[x * 16 + 7 + 8] = C[x * 16 + 7 + 8] + A[x * 16 + 7 + 8] * B[x * 16 + 7 + 8];
            }
        }
    }
}


void vector_kernel() {
    #pragma omp parallel for
    for (int i = 0; i < 10240; ++i) {
        __m512 A[5];
        __m512 B[5];
        __m512 C[5];
        for (int x = 0; x < 5; ++x) {
            A[x] = _mm512_set1_ps(0.0f);
            B[x] = _mm512_set1_ps(1.0f);
            C[x] = _mm512_set1_ps(0.0f);
        }
        for (int j = 0; j < 1024; ++j) {
            __m512 tmp = _mm512_mul_ps(A[0], B[0]);
            C[0] = _mm512_add_ps(tmp, C[0]);
            tmp = _mm512_mul_ps(A[1], B[1]);
            C[1] = _mm512_add_ps(tmp, C[1]);

            tmp = _mm512_mul_ps(A[2], B[2]);
            C[2] = _mm512_add_ps(tmp, C[2]);
            tmp = _mm512_mul_ps(A[3], B[3]);
            C[3] = _mm512_add_ps(tmp, C[3]);

            tmp = _mm512_mul_ps(A[4], B[4]);
            C[4] = _mm512_add_ps(tmp, C[4]);
        }
    }
}


void hybrid_kernel() {
    #pragma omp parallel for
    for (int i = 0; i < 10240; ++i) {
        __m512 A[4];
        __m512 B[4];
        __m512 C[4];
        for (int x = 0; x < 4; ++x) {
            A[x] = _mm512_set1_ps(0.0f);
            B[x] = _mm512_set1_ps(1.0f);
            C[x] = _mm512_set1_ps(0.0f);
        }
        float AA[16] = {0.0f};
        float BB[16] = {1.0f};
        float CC[16] = {0.0f};
        for (int j = 0; j < 1024; ++j) {
            __m512 tmp = _mm512_mul_ps(A[0], B[0]);
            CC[0] = CC[0] + AA[0] * BB[0];
            CC[1] = CC[1] + AA[1] * BB[1];
            C[0] = _mm512_add_ps(tmp, C[0]);
            CC[2] = CC[2] + AA[2] * BB[2];
            CC[3] = CC[3] + AA[3] * BB[3];
            tmp = _mm512_mul_ps(A[1], B[1]);
            CC[4 + 0] = CC[4 + 0] + AA[4 + 0] * BB[4 + 0];
            CC[4 + 1] = CC[4 + 1] + AA[4 + 1] * BB[4 + 1];
            C[1] = _mm512_add_ps(tmp, C[1]);
            CC[4 + 2] = CC[4 + 2] + AA[4 + 2] * BB[4 + 2];
            CC[4 + 3] = CC[4 + 3] + AA[4 + 3] * BB[4 + 3];

            tmp = _mm512_mul_ps(A[2], B[2]);
            CC[8 + 0] = CC[8 + 0] + AA[8 + 0] * BB[8 + 0];
            CC[8 + 1] = CC[8 + 1] + AA[8 + 1] * BB[8 + 1];
            C[2] = _mm512_add_ps(tmp, C[2]);
            CC[8 + 2] = CC[8 + 2] + AA[8 + 2] * BB[8 + 2];
            CC[8 + 3] = CC[8 + 3] + AA[8 + 3] * BB[8 + 3];
            tmp = _mm512_mul_ps(A[3], B[3]);
            CC[12 + 0] = CC[12 + 0] + AA[12 + 0] * BB[12 + 0];
            CC[12 + 1] = CC[12 + 1] + AA[12 + 1] * BB[12 + 1];
            C[3] = _mm512_add_ps(tmp, C[3]);
            CC[12 + 2] = CC[12 + 2] + AA[12 + 2] * BB[12 + 2];
            CC[12 + 3] = CC[12 + 3] + AA[12 + 3] * BB[12 + 3];

        }
    }
}


int main(int argc, char *argv[])
{

    float fma_time, vector_time, hybrid_time;
    {
        // warm up
        fma_kernel();
        std::chrono::high_resolution_clock::time_point t1;
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; ++i) {
            fma_kernel();
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        fma_time = float(time_span.count() / repeat * 1e3);
    }

    {
        // warm up
        vector_kernel();
        std::chrono::high_resolution_clock::time_point t1;
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; ++i) {
            vector_kernel();
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        vector_time = float(time_span.count() / repeat * 1e3);
    }

    {
        // warm up
        hybrid_kernel();
        std::chrono::high_resolution_clock::time_point t1;
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeat; ++i) {
            hybrid_kernel();
        }
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        hybrid_time = float(time_span.count() / repeat * 1e3);
    }

    std::cout << "FMA: " << fma_time << "ms, Vector: " << vector_time << "ms, Hybrid: " << hybrid_time << "ms, Relative Perf: " << fma_time / hybrid_time << "\n"
                << std::flush;

    return 0;
}