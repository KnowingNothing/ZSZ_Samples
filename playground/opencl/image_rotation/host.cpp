#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <chrono>
#include <cmath>

#define PLATFORM 0
#define DEVICE 0
#define M 1024
#define N M

#define KERNEL "kernel.cl"
#define NAME "image_rotation"
#define THETA 45

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


void cpu_image_rotate(float* input, float* output, int height, int width, float theta) {
    int i, j;
    int xc = height / 2;
    int yc = width / 2;
    float pi = 3.1415926;

    for(i = 0; i < height; ++i) {
        for(j = 0; j < width; ++j) {
            int xpos =  (i - xc) * cos(pi / 180 * theta) - (j - yc) * sin(pi / 180 * theta) + xc;   
            int ypos =  (i - xc) * sin(pi / 180 * theta) + (j - yc) * cos(pi / 180 * theta) + yc;

            if(xpos >= 0 && ypos >= 0 && xpos < width && ypos < height)
                output[xpos * width + ypos] = input[i * width + j];
        }
    }
}


int main() {
    size_t image_size = M * N;
    const int platform_id = PLATFORM;
    const int device_id = DEVICE;

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

    float* host_input;
    float* host_output;

    std::cout << "Initializing inputs...\n" << std::flush; 
    host_input = (float*)malloc(image_size * 4 * sizeof(float));
    host_output = (float*)malloc(image_size * 4 * sizeof(float));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int c = 0; c < 4; ++c) {
                host_input[i * N + j * 4 + c] = ((i + j * 29.0) / 791 + 2) / 8;
                host_output[i * N + j * 4 + c] = 0;
            }
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
    
    cl_mem device_input;  // Input array on the device
    cl_mem device_output;  // Input array on the device
    cl_image_desc  desc;
    cl_image_format format;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {M, N, 1};

    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = N;
    desc.image_height = M;
    desc.image_depth = 0;
    desc.image_array_size = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;

    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_FLOAT;

    // Use clCreateBuffer() to create a buffer object (d_A) 
    // that will contain the data from the host array A
    std::cout << "Creating buffer...\n" << std::flush; 
    device_input = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, NULL);
    device_output = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, NULL);

    std::cout << "Load data to device...\n" << std::flush; 
    // Use clEnqueueWriteBuffer() to write input array A to
    // the device buffer bufferA
    status = clEnqueueWriteImage(
        cmdQueue, 
        device_input, 
        CL_TRUE, 
        origin, 
        region,
        0,
        0,
        host_input,
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
    kernel = clCreateKernel(program, NAME, &status);

    std::cout << "Set arguments...\n" << std::flush; 
    
    // Associate the input and output buffers with the 
    // kernel 
    // using clSetKernelArg()
    int height = M;
    int width = N;
    float theta = THETA;
    status  = clSetKernelArg(
        kernel, 
        0, 
        sizeof(cl_mem), 
        &device_input);
    CHECK_CL(status, "Can't assign args0");
    status |= clSetKernelArg(
        kernel, 
        1, 
        sizeof(cl_mem), 
        &device_output);
    CHECK_CL(status, "Can't assign args1");
    status |= clSetKernelArg(
        kernel, 
        2, 
        sizeof(cl_int), 
        &height);
    CHECK_CL(status, "Can't assign args2");
    status |= clSetKernelArg(
        kernel, 
        3, 
        sizeof(cl_int), 
        &width);
    CHECK_CL(status, "Can't assign args3");
    status |= clSetKernelArg(
        kernel, 
        4, 
        sizeof(cl_float), 
        &theta);
    CHECK_CL(status, "Can't assign args4");

    size_t globalWorkSize[2];    
    // There are 'elements' work-items 
    globalWorkSize[0] = M;
    globalWorkSize[1] = N;

    // size_t localWorkSize[2];
    // localWorkSize[0] = 8;
    // localWorkSize[1] = 8;

    std::cout << "Launch kernel...\n" << std::flush; 
    status = clEnqueueNDRangeKernel(
        cmdQueue, 
        kernel, 
        2, 
        NULL, 
        globalWorkSize, 
        NULL, 
        0, 
        NULL, 
        NULL);

    CHECK_CL(status, "Can't enqueue kernel");

    std::cout << "Fetch result...\n" << std::flush; 
    clEnqueueReadImage(
        cmdQueue, 
        device_output, 
        CL_TRUE, 
        origin,
        region,
        0,
        0, 
        host_output, 
        0, 
        NULL, 
        NULL);

    // measure time
    std::cout << "Measure time...\n" << std::flush; 
    cl_event event_start, event_end;
    int repeat = 20;
    for (int i = 0; i <= 20; ++i) {
        if (i == 1) {
            status = clEnqueueNDRangeKernel(
                            cmdQueue, 
                            kernel, 
                            2, 
                            NULL, 
                            globalWorkSize, 
                            NULL, 
                            0, 
                            NULL, 
                            &event_start);
        } else if (i == 20) {
            status = clEnqueueNDRangeKernel(
                            cmdQueue, 
                            kernel, 
                            2, 
                            NULL, 
                            globalWorkSize, 
                            NULL, 
                            0, 
                            NULL, 
                            &event_end);
        } else {
            status = clEnqueueNDRangeKernel(
                            cmdQueue, 
                            kernel, 
                            2, 
                            NULL, 
                            globalWorkSize, 
                            NULL, 
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
    std::cout << "Time cost for M=" << M << ", N=" << N << " is "
              << duration.count() / repeat / 1e6 << " ms.\n";

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        cpu_image_rotate(host_input, host_output, M, N, THETA);
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "CPU serial cost is " << time_span.count() / repeat * 1e3 << " ms.\n";


    std::cout << "Clean up...\n" << std::flush; 
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(device_input);
    clReleaseMemObject(device_output);
    clReleaseContext(context);

    // Free host resources
    free(host_input);
    free(host_output);
    free(kernel_source);

    std::cout << "Done!\n" << std::flush; 
    return 0;
}