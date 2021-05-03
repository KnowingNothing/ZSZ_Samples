#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <chrono>

// #define USE_FP64
// #define CHECK_CORRECTNESS

#if defined(USE_FP64)
#define CLDType cl_double
#define DType double
#define DTypeBits 64
#define KERNEL "gemm_pipeline_opt_fp64.cl"
#else
#define CLDType cl_float
#define DType float
#define DTypeBits 32
#define KERNEL "gemm_pipeline_opt_fp32.cl"

#endif

#define PLATFORM 0
#define DEVICE 0
#define M (8192)
#define N M
#define K M

#define CEIL(a, b) ((a + b - 1) / b)

#define BM 4
#define BN 4
#define BK 4


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

    cl_uint num_platforms;
    CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms), "Can't get platform ids");
    cl_platform_id platforms[num_platforms];
    CHECK_CL(clGetPlatformIDs(num_platforms, platforms, NULL), "Can't get platforms");
    if (num_platforms <= platform_id) {
        std::cerr << "Can't get expected platform: " << platform_id
                  << " within " << num_platforms << " platforms.\n";
        abort();
    }

    cl_platform_id platform = platforms[platform_id];
    cl_uint num_devices;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices), "Can't get device number.");
    cl_device_id devices[num_devices];
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL), "Can't get device ids.");
    if (num_devices <= device_id) {
        std::cerr << "Can't get expected device: " << device_id
                  << " within " << num_devices << " devices.\n";
        abort();
    }

    cl_device_id device = devices[device_id];
#if defined(USE_FP64)
    check_fp64_capability(device);
#endif

    std::cout << "Loading kernel source...\n" << std::flush; 
    std::ifstream kernel_src(KERNEL, std::ifstream::in);
    if (!kernel_src.good()) {
        std::cerr << "Can't open source file: " << KERNEL << "\n";
        abort();
    }

    char* kernel_source;
    kernel_src.seekg(0, kernel_src.end);
    int kernel_size = kernel_src.tellg();
    kernel_source = (char*)malloc(kernel_size);
    kernel_src.seekg(0, kernel_src.beg);
    kernel_src.read(kernel_source, kernel_size);
    
    kernel_src.close();

    DType *host_A, *host_B, *host_C, *host_golden;
    host_A = (DType*)malloc(A_bytes);
    host_B = (DType*)malloc(B_bytes);
    host_C = (DType*)malloc(C_bytes);
    host_golden = (DType*)malloc(C_bytes);
    std::cout << "Initializing inputs...\n" << std::flush; 
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            host_A[i * K + k] = ((i + k * 29) / 791 + 2) / 8 % 2;
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            host_B[j * K + k] = ((j * 97 + k) / 111 + 951) % 2;
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            host_C[i * N + j] = 0;
            host_golden[i * N + j] = 0;
        }
    }

    cl_context context = NULL;

    // Create a context using clCreateContext() and 
    // associate it with the devices
    cl_int status;
    context = clCreateContext(
        NULL, 
        num_devices, 
        devices, 
        NULL, 
        NULL, 
        &status);

    cl_command_queue_properties properties[] {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, devices[0], properties, &status);
    
    cl_mem bufferA;  // Input array on the device
    cl_mem bufferB;  // Input array on the device
    cl_mem bufferC;  // Output array on the device

    // Use clCreateBuffer() to create a buffer object (d_A) 
    // that will contain the data from the host array A
    std::cout << "Creating buffer...\n" << std::flush; 
    bufferA = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        A_bytes, 
        NULL, 
        &status);

    // Use clCreateBuffer() to create a buffer object (d_B)
    // that will contain the data from the host array B
    bufferB = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        B_bytes, 
        NULL, 
        &status);

    // Use clCreateBuffer() to create a buffer object (d_C) 
    // with enough space to hold the output data
    bufferC = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,                 
        C_bytes, 
        NULL, 
        &status);

    std::cout << "Load data to device...\n" << std::flush; 
    // Use clEnqueueWriteBuffer() to write input array A to
    // the device buffer bufferA
    status = clEnqueueWriteBuffer(
        cmdQueue, 
        bufferA, 
        CL_FALSE, 
        0, 
        A_bytes,                         
        host_A, 
        0, 
        NULL, 
        NULL);
    
    // Use clEnqueueWriteBuffer() to write input array B to 
    // the device buffer bufferB
    status = clEnqueueWriteBuffer(
        cmdQueue, 
        bufferB, 
        CL_FALSE, 
        0, 
        B_bytes,                                  
        host_B, 
        0, 
        NULL, 
        NULL);

    std::cout << "Build program...\n" << std::flush; 
    // Create a program using clCreateProgramWithSource()
    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**)&(kernel_source),                                 
        NULL, 
        &status);

    // Build (compile) the program for the devices with
    // clBuildProgram()
    status = clBuildProgram(
        program, 
        num_devices, 
        devices, 
        NULL, 
        NULL, 
        NULL);

    cl_kernel kernel = NULL;

    std::cout << "Create kernel...\n" << std::flush; 
    // Use clCreateKernel() to create a kernel from the 
    // vector addition function (named "vecadd")
    kernel = clCreateKernel(program, "gemm_opt", &status);

    std::cout << "Set arguments...\n" << std::flush; 
    
    // Associate the input and output buffers with the 
    // kernel 
    // using clSetKernelArg()
    cl_int stride_A = K;
    cl_int stride_B = K;
    cl_int stride_C = N;
    cl_int MM = M;
    cl_int NN = N;
    cl_int KK = K;
    status  = clSetKernelArg(
        kernel, 
        0, 
        sizeof(cl_mem), 
        &bufferA);
    CHECK_CL(status, "Can't assign args0");
    status |= clSetKernelArg(
        kernel, 
        1, 
        sizeof(cl_mem), 
        &bufferB);
    CHECK_CL(status, "Can't assign args1");
    status |= clSetKernelArg(
        kernel, 
        2, 
        sizeof(cl_mem), 
        &bufferC);
    CHECK_CL(status, "Can't assign args2");
    status |= clSetKernelArg(
        kernel, 
        3, 
        sizeof(cl_int), 
        &stride_A);
    CHECK_CL(status, "Can't assign args3");
    status |= clSetKernelArg(
        kernel, 
        4, 
        sizeof(cl_int), 
        &stride_B);
    CHECK_CL(status, "Can't assign args4");
    status |= clSetKernelArg(
        kernel, 
        5, 
        sizeof(cl_int), 
        &stride_C);
    CHECK_CL(status, "Can't assign args5");
    status |= clSetKernelArg(
        kernel, 
        6, 
        sizeof(cl_int), 
        &MM);
    CHECK_CL(status, "Can't assign args3");
    status |= clSetKernelArg(
        kernel, 
        7, 
        sizeof(cl_int), 
        &NN);
    CHECK_CL(status, "Can't assign args4");
    status |= clSetKernelArg(
        kernel, 
        8, 
        sizeof(cl_int), 
        &KK);
    CHECK_CL(status, "Can't assign args5");

    size_t groups[2];    
    // There are 'elements' work-items 
    groups[0] = CEIL(M, BM) * BM / 8;
    groups[1] = CEIL(N, BN) * BN / 8;

    std::cout << groups[0] << " " << groups[1] << "\n";
    
    size_t locals[2];
    locals[0] = BM;
    locals[1] = BN;

    std::cout << locals[0] << " " << locals[1] << "\n";

    std::cout << "Launch kernel...\n" << std::flush; 
    status = clEnqueueNDRangeKernel(
        cmdQueue, 
        kernel, 
        2, 
        NULL, 
        groups, 
        locals, 
        0, 
        NULL, 
        NULL);

    CHECK_CL(status, "Can't enqueue kernel");

    std::cout << "Fetch result...\n" << std::flush; 
    clEnqueueReadBuffer(
        cmdQueue, 
        bufferC, 
        CL_TRUE, 
        0, 
        C_bytes, 
        host_C, 
        0, 
        NULL, 
        NULL);

#if defined(CHECK_CORRECTNESS)
    std::cout << "Check correctness...\n" << std::flush; 
    int errors = 0;
    for (int i = 0; i < CEIL(M, 16); ++i) {
        for (int j = 0; j < CEIL(N, 16); ++j) {
            for (int k = 0; k < CEIL(K, 16); ++k) {
                for (int ii = 0; ii < 16; ++ii) {
                    int m = i * 16 + ii;
                    for (int kk = 0; kk < 16 && m < M; ++kk) {
                        int kx = k * 16 + kk;
                        for (int jj = 0; jj < 16 && kx < K; ++jj) {
                            int n = j * 16 + jj;
                            if (n < N) {
                                host_golden[m * N + n] += host_A[m * K + kx] * host_B[n + kx * N];
                            }
                        }
                    }
                }
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if (std::abs(host_golden[m * N + n] - host_C[m * N + n]) > 1e-1) {
                std::cout << host_golden[m * N + n] << " vs " << host_C[m * N + n] << "\n";
                errors += 1;
            }
        }
    }
    if (errors > 0) {
        std::cerr << "Compuet incorrect! Find " << errors << " errors.\n";
        abort();
    }
#endif

    // measure time
    std::cout << "Measure time...\n" << std::flush; 
    cl_event event_start, event_end;
    int repeat = 20;
    for (int i = 0; i <= 20; ++i) {
        if (i == 1) {
            status = clEnqueueNDRangeKernel(
                            cmdQueue, 
                            kernel, 
                            1, 
                            NULL, 
                            groups, 
                            locals, 
                            0, 
                            NULL, 
                            &event_start);
        } else if (i == 20) {
            status = clEnqueueNDRangeKernel(
                            cmdQueue, 
                            kernel, 
                            1, 
                            NULL, 
                            groups, 
                            locals, 
                            0, 
                            NULL, 
                            &event_end);
        } else {
            status = clEnqueueNDRangeKernel(
                            cmdQueue, 
                            kernel, 
                            1, 
                            NULL, 
                            groups, 
                            locals, 
                            0, 
                            NULL, 
                            NULL);
        }
    }

    clWaitForEvents(1, &event_end);
    cl_ulong start, end;
    clGetEventProfilingInfo(event_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    clGetEventProfilingInfo(event_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);

    std::chrono::nanoseconds duration{end - start};
    std::cout << "Time cost for M=" << M << ", N=" << N << ", K=" << K << " is "
              << duration.count() / repeat / 1e6 << " ms.\n";

    std::cout << "Clean up...\n" << std::flush; 
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);

    // Free host resources
    free(host_A);
    free(host_B);
    free(host_C);
    free(host_golden);
    free(kernel_source);

    std::cout << "Done!\n" << std::flush; 
    return 0;
}