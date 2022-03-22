#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <chrono>
#include <immintrin.h>

// #define USE_FP64
// #define CHECK_CORRECTNESS

#if defined(USE_FP64)
#define CLDType cl_double
#define DType double
#define DTypeBits 64
#define KERNEL "gemm_naive_fp64.cl"
#else
#define CLDType cl_float
#define DType float
#define DTypeBits 32
#define KERNEL "gemm_naive_fp32.cl"

#endif

#define PLATFORM 0
#define DEVICE 0
#define M 8192
#define N M
#define K M

#define CEIL(a, b) ((a + b - 1) / b)


std::string status2str(cl_int status) {
  if (status == CL_SUCCESS) {
    return "CL_SUCCESS";
  } else if (status == CL_INVALID_VALUE) {
    return "CL_​INVALID_​VALUE";
  } else if (status == CL_OUT_OF_HOST_MEMORY) {
    return "CL_​OUT_​OF_​HOST_​MEMORY";
  } else {
    throw std::runtime_error("Unknown status: " + std::to_string(status));
  }
}


#define CHECK_CL(status, str)                                                  \
  if (status != CL_SUCCESS) {                                                  \
    std::cout << "OpenCL Failure: " << str << status2str(status) << "\n"       \
              << std::flush;                                                   \
    exit(1);                                                                   \
  }


std::string get_device_info(cl_device_id device, cl_device_info params) {
  size_t buffer_size;
  CHECK_CL(clGetDeviceInfo(device, params, 0, NULL, &buffer_size), "Can't get device info buffer size.");
  char* buffer = (char*)malloc(buffer_size);
  CHECK_CL(clGetDeviceInfo(device, params, buffer_size, buffer, NULL), "Can't get device info.");
  std::string ret = std::string(buffer);
  if (params == CL_DEVICE_MAX_COMPUTE_UNITS
      || params == CL_DEVICE_MAX_CLOCK_FREQUENCY
      || params == CL_DEVICE_DOUBLE_FP_CONFIG) {
    cl_uint num = *((cl_uint*)buffer);
    std::ostringstream oss;
    oss << num;
    ret = oss.str();
  } else if (params == CL_DEVICE_GLOBAL_MEM_SIZE) {
    cl_ulong num = *((cl_ulong*)buffer);
    num /= 1e9;
    std::ostringstream oss;
    oss << num;
    ret = oss.str();
  }
  free(buffer);
  return ret;
}


void check_fp64_capability(cl_device_id device) {
    std::string cfg = get_device_info(device, CL_DEVICE_DOUBLE_FP_CONFIG);
    int value = std::atoi(cfg.c_str());
    if (value == 0) {
        std::cerr << "Not support for FP64 found!.\n";
        abort();
    }
}


template<int BM, int BN, int BK>
void gemm_opt(float* A, float* B, float* C) {
    #define B_REG_M 2
    __m256 C_vec[B_REG_M * BN / 8];

    for (int i = 0; i < BM; i += B_REG_M) {
        for (int k = 0; k < B_REG_M * BN / 8; ++k) {
            C_vec[k] = _mm256_setzero_ps();
        }

        for (int k = 0; k < BK; ++k) {
            __m256 B_vec[BN / 8];
            for (int jj = 0; jj < BN / 8; ++jj) {
                B_vec[jj] = _mm256_loadu_ps(B + k * BN + jj * 8);
            }

            for (int ii = 0; ii < B_REG_M; ++ii) {
                __m256 A_vec = _mm256_broadcast_ss(A + (i + ii) * BK + k);

                for (int jj = 0; jj < BN / 8; ++jj) {
                    __m256 tmp = _mm256_mul_ps(A_vec, B_vec[jj]);
                    C_vec[ii * (BN/8) + jj] = _mm256_add_ps(tmp, C_vec[ii * (BN/8) + jj]);
                }
            }
        }

        for (int ii = 0; ii < B_REG_M; ++ii) {
            for (int jj = 0; jj < BN / 8; ++jj) {
                _mm256_storeu_ps(C + (i + ii) * BN + jj * 8, C_vec[ii * (BN / 8) + jj]);
            }
        }

    }
    #undef B_REG_M
}


int main() {
    const int A_dtype_bytes = DTypeBits / 8;
    const int B_dtype_bytes = DTypeBits / 8;
    const int C_dtype_bytes = DTypeBits / 8;
    const int platform_id = PLATFORM;
    const int device_id = DEVICE;

    int A_elements = M * K;
    int B_elements = N * K;
    int C_elements = M * N;

    int A_bytes = A_elements * A_dtype_bytes;
    int B_bytes = B_elements * B_dtype_bytes;
    int C_bytes = C_elements * C_dtype_bytes;

    DType *host_A, *host_B, *host_C, *host_golden;
    host_A = (DType*)malloc(A_bytes);
    host_B = (DType*)malloc(B_bytes);
    host_C = (DType*)malloc(C_bytes);
    std::cout << "Initializing inputs...\n" << std::flush; 
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            host_A[i * K + k] = ((i + k * 29.0) / 791 + 2) / 8;
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            host_B[j * K + k] = ((j * 97.0 + k) / 111 + 951) * 3.14;
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            host_C[i * N + j] = 0;
        }
    }

    // measure time
    std::cout << "Measure time...\n" << std::flush; 
    cl_event event_start, event_end;
    int repeat = 20;
    std::chrono::high_resolution_clock::time_point t1;
    for (int i = 0; i <= 20; ++i) {
        if (i == 1) {
            t1 = std::chrono::high_resolution_clock::now();
            gemm_opt<M, N, K>(host_A, host_B, host_C);
        } else if (i == 20) {
            gemm_opt<M, N, K>(host_A, host_B, host_C);
        } else {
            gemm_opt<M, N, K>(host_A, host_B, host_C);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "CPU time cost for M=" << M << ", N=" << N << ", K=" << K << " is "
              << time_span.count() / repeat * 1e3 << " ms.\n";

    // Free host resources
    free(host_A);
    free(host_B);
    free(host_C);

    std::cout << "Done!\n" << std::flush; 
    return 0;
}