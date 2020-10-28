#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }

void cudaErrCheck_(cudaError_t stat, const char* file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char* file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char* file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}


#include <mma.h>
using namespace nvcuda;

#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__global__ void wmma_fp16(half* a, half* b, float* c, int M, int N, int K, float alpha, float beta) {
  int lda = K;
  int ldb = N;
  int ldc = N;

  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
      wmma::load_matrix_sync(b_frag, b + bRow * lda + bCol, ldb);

      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; ++i) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }
    wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
  }
}


__global__ void convertFp32ToFp16 (half* out, float* in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < n; i += stride) {
    out[i] = (half)in[i];
    // out[i] = half(i % 100);
  }
}


__global__ void transpose(half* dst, half* src, int contiguous, int strided) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int n = contiguous * strided;
  for (int i = idx; i < n; i += stride) {
    int d_strided = i % contiguous;
    int d_contiguous = i / contiguous;
    int d_i = d_strided * strided + d_contiguous;
    dst[d_i] = src[i];
  }
}


int main(int argc, char* argv[]) {
  float* a_fp32;
  float* b_fp32;
  half* a_fp16;
  half* b_fp16;
  // half* a_row_fp16;
  // half* b_row_fp16;

  float* c;
  float* c_cublas;
  float* c_wmma;

  // half* a_host;
  // half* b_host;
  float* c_host_cublas;
  float* c_host_wmma;

  cudaStream_t stream1 = 0, stream2 = 0;
  cudaErrCheck(cudaStreamCreate(&stream1));
  cudaErrCheck(cudaStreamCreate(&stream2));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  cudaEvent_t startWMMA;
  cudaEvent_t stopWMMA;
  
  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;

  cudaErrCheck(cudaEventCreate(&startWMMA));
  cudaErrCheck(cudaEventCreate(&stopWMMA));

  cudaErrCheck(cudaEventCreate(&startcublas));
  cudaErrCheck(cudaEventCreate(&stopcublas));

  cublasErrCheck(cublasCreate(&cublasHandle));

  cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_N * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_N * MATRIX_K * sizeof(half)));
  // cudaErrCheck(cudaMalloc((void**)&a_row_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  // cudaErrCheck(cudaMalloc((void**)&b_row_fp16, MATRIX_N * MATRIX_K * sizeof(half)));

  cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

  // a_host = (half*)malloc(MATRIX_M * MATRIX_K * sizeof(half));
  // b_host = (half*)malloc(MATRIX_N * MATRIX_K * sizeof(half));
  c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
  c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

  curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
  curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_N * MATRIX_K));

  convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256, 0, stream1>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
  convertFp32ToFp16<<<(MATRIX_N * MATRIX_K + 255) / 256, 256, 0, stream2>>>(b_fp16, b_fp32, MATRIX_N * MATRIX_K);

  // cudaErrCheck(cudaMemcpy(a_host, a_fp16, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyDeviceToHost));
  // cudaErrCheck(cudaMemcpy(b_host, b_fp16, MATRIX_N * MATRIX_K * sizeof(half), cudaMemcpyDeviceToHost));
  // printf("\nA matrix before layout transpose:\n");
  // for (int i = 0; i < MATRIX_M; ++i) {
  //   for (int j = 0; j < MATRIX_K; ++j) {
  //     printf("%f ", float(a_host[i + j * MATRIX_M]));
  //   }
  //   printf("\n");
  // }
  // printf("\nB matrix before layout transpose:\n");
  // for (int i = 0; i < MATRIX_K; ++i) {
  //   for (int j = 0; j < MATRIX_N; ++j) {
  //     printf("%f ", float(b_host[i + j * MATRIX_K]));
  //   }
  //   printf("\n");
  // }

  // transpose<<<(MATRIX_M * MATRIX_K + 255) / 256, 256, 0, stream1>>>(a_row_fp16, a_fp16, MATRIX_M, MATRIX_K);
  // transpose<<<(MATRIX_N * MATRIX_K + 255) / 256, 256, 0, stream2>>>(b_row_fp16, b_fp16, MATRIX_K, MATRIX_N);

  // cudaErrCheck(cudaMemcpy(a_host, a_row_fp16, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyDeviceToHost));
  // cudaErrCheck(cudaMemcpy(b_host, b_row_fp16, MATRIX_N * MATRIX_K * sizeof(half), cudaMemcpyDeviceToHost));
  // printf("\nA matrix after layout transpose:\n");
  // for (int i = 0; i < MATRIX_M; ++i) {
  //   for (int j = 0; j < MATRIX_K; ++j) {
  //     printf("%f ", float(a_host[i * MATRIX_K + j]));
  //   }
  //   printf("\n");
  // }
  // printf("\nB matrix after layout transpose:\n");
  // for (int i = 0; i < MATRIX_K; ++i) {
  //   for (int j = 0; j < MATRIX_N; ++j) {
  //     printf("%f ", float(b_host[i * MATRIX_N + j]));
  //   }
  //   printf("\n");
  // }

  curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_K));
  curandErrCheck(curandDestroyGenerator(gen));

  cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
  cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

  float alpha = 2.0f;
  float beta = 2.0f;

  printf("\nM = %d, N = %d, K = %d, alpha=%f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;
  int WarpSize = 32;
  gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / WarpSize - 1) / (WMMA_M * blockDim.x / WarpSize));
  gridDim.y = (MATRIX_M + (WMMA_N * blockDim.y - 1)) / (WMMA_N * blockDim.y);

  printf("Running with wmma...\n");
  cudaErrCheck(cudaEventRecord(startWMMA));
  wmma_fp16<<<gridDim, blockDim, 0, stream1>>>(a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  cudaErrCheck(cudaEventRecord(stopWMMA));

  printf("Running with cuBLAS...\n");
  cudaErrCheck(cudaEventRecord(startcublas));
  cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              MATRIX_N, MATRIX_M, MATRIX_K,
                              &alpha,
                              b_fp16, CUDA_R_16F, MATRIX_N,
                              a_fp16, CUDA_R_16F, MATRIX_K,
                              &beta,
                              c_cublas, CUDA_R_32F, MATRIX_N,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                            ));
  cudaErrCheck(cudaEventRecord(stopcublas));

  printf("\nChecking results...\n");
  cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  
  // printf("\nResults of WMMA:\n");
  // for (int i = 0; i < MATRIX_M; ++i) {
  //   for (int j = 0; j < MATRIX_N; ++j) {
  //     printf("%f ", c_host_wmma[i + j * MATRIX_M]);
  //   }
  //   printf("\n");
  // }
  // for (int i = 0; i < MATRIX_M * MATRIX_N; ++i)
  //   printf("%f ", c_host_wmma[i]);
  // printf("\n");

  // printf("\nResults of cuBLAS:\n");
  // for (int i = 0; i < MATRIX_M; ++i) {
  //   for (int j = 0; j < MATRIX_N; ++j) {
  //     printf("%f ", c_host_cublas[i * MATRIX_N + j]);
  //   }
  //   printf("\n");
  // }
  // for (int i = 0; i < MATRIX_M * MATRIX_N; ++i)
  //   printf("%f ", c_host_cublas[i]);
  // printf("\n");
  
  int errors = 0;
  for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
    float v1 = c_host_wmma[i];
    float v2 = c_host_cublas[i];
    if (v1 / v2 > 1.001 || v2 / v1 > 1.001) {
      errors++;
      if (errors < 10) printf("%f %f\n", v1, v2);
    }
  }
  if (errors > 0) {
    printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
  }
  else {
    printf("Results verified: cublas and WMMA agree.\n\n");
    float wmmaTime;
    float cublasTime;
    cudaErrCheck(cudaEventSynchronize(stopWMMA));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
    printf("wmma took %fms\n", wmmaTime);
    printf("cublas took %fms\n", cublasTime);

    printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
  }
  
  
  cudaErrCheck(cudaEventDestroy(startWMMA));
  cudaErrCheck(cudaEventDestroy(stopWMMA));

  cudaErrCheck(cudaEventDestroy(startcublas));             
  cudaErrCheck(cudaEventDestroy(stopcublas));
  
  cudaErrCheck(cudaFree(a_fp32));
  cudaErrCheck(cudaFree(b_fp32));
  cudaErrCheck(cudaFree(a_fp16));
  cudaErrCheck(cudaFree(b_fp16));

  cudaErrCheck(cudaFree(c));
  cudaErrCheck(cudaFree(c_cublas));
  cudaErrCheck(cudaFree(c_wmma));
  
  free(c_host_cublas);
  free(c_host_wmma);

  cudaErrCheck(cudaDeviceReset());
  return 0;
}