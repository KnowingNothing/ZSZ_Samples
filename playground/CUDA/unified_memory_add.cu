#include <iostream>
#include <cmath>
#include "cuda_runtime.h"


#define checkCUDA(expression)                               \
  {                                                         \
    cudaError_t status = (expression);                      \
    if (status != cudaSuccess) {                            \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl; \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  }
  


__global__ void add(int n, float* x, float* y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}


int main(void) {
  int N = 1 << 20;
  float *x, *y;

  checkCUDA(cudaSetDevice(0));
  cudaStream_t stream = 0;

  checkCUDA(cudaMallocManaged(&x, N*sizeof(float)));
  checkCUDA(cudaMallocManaged(&y, N*sizeof(float)));

  for (int i = 0; i < N; ++i) {
    x[i] = float(std::sin(i));
    y[i] = float(-std::sin(i));
  }

  dim3 block(256, 1, 1);
  dim3 grid((N + 255) / 256, 1, 1);
  add<<<grid, block, 0, stream>>>(N, x, y);

  checkCUDA(cudaDeviceSynchronize());
  float max_error = 0.0;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(y[i]) > 1e-5) {
      max_error = std::max(max_error, std::fabs(y[i]));
    }
  }
  std::cout << "Max error: " << max_error << "\n";

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(y));
  return 0;
}

// compile: nvcc ./unified_memory_add.cu
// profile: ncu ./a.out
// Results by Nsight on A100:
// Max error: 0
// ==PROF== Disconnected from process 14142
// [14142] a.out@127.0.0.1
//   add(int, float*, float*), 2020-Oct-28 10:36:31, Context 1, Stream 7
//     Section: GPU Speed Of Light
//     ---------------------------------------------------------------------- --------------- ------------------------------
//     DRAM Frequency                                                           cycle/nsecond                           1.11
//     SM Frequency                                                             cycle/usecond                         700.12
//     Elapsed Cycles                                                                   cycle                          9,729
//     Memory [%]                                                                           %                          47.36
//     SOL DRAM                                                                             %                          42.37
//     Duration                                                                       usecond                          13.89
//     SOL L1/TEX Cache                                                                     %                          24.11
//     SOL L2 Cache                                                                         %                          60.91
//     SM Active Cycles                                                                 cycle                       7,198.41
//     SM [%]                                                                               %                          15.26
//     ---------------------------------------------------------------------- --------------- ------------------------------
//     WRN   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance
//           of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate
//           latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

//     Section: Launch Statistics
//     ---------------------------------------------------------------------- --------------- ------------------------------
//     Block Size                                                                                                        256
//     Grid Size                                                                                                       4,096
//     Registers Per Thread                                                   register/thread                             16
//     Shared Memory Configuration Size                                                 Kbyte                          32.77
//     Driver Shared Memory Per Block                                             Kbyte/block                           1.02
//     Dynamic Shared Memory Per Block                                             byte/block                              0
//     Static Shared Memory Per Block                                              byte/block                              0
//     Threads                                                                         thread                      1,048,576
//     Waves Per SM                                                                                                     4.74
//     ---------------------------------------------------------------------- --------------- ------------------------------
//     WRN   A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the
//           target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical
//           occupancy of the kernel. This kernel launch results in 4 full waves and a partial wave of 639 thread blocks.
//           Under the assumption of a uniform execution duration of all thread blocks, the partial wave may account for
//           up to 20.0% of the total kernel runtime with a lower occupancy of 26.5%. Try launching a grid with no
//           partial wave. The overall impact of this tail effect also lessens with the number of full waves executed for
//           a grid.

//     Section: Occupancy
//     ---------------------------------------------------------------------- --------------- ------------------------------
//     Block Limit SM                                                                   block                             32
//     Block Limit Registers                                                            block                             16
//     Block Limit Shared Mem                                                           block                            164
//     Block Limit Warps                                                                block                              8
//     Theoretical Active Warps per SM                                                   warp                             64
//     Theoretical Occupancy                                                                %                            100
//     Achieved Occupancy                                                                   %                          73.53
//     Achieved Active Warps Per SM                                                      warp                          47.06
//     ---------------------------------------------------------------------- --------------- ------------------------------
