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

#define KERNEL "kernel.cl"
#define NAME1 "test1"
#define NAME2 "test2"


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


int main() {
    size_t buffer_size = 1024 * 1024 * 4 * sizeof(float);
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

    std::cout << "Loading kernel1 source...\n" << std::flush; 
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
    host_input = (float*)malloc(buffer_size);
    host_output = (float*)malloc(buffer_size);

    for (int i = 0; i < buffer_size / (sizeof(float)); ++i) {
        host_input[i] = i;
        host_output[i] = i + 1;
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
    
    cl_mem device_input1;  // Input array on the device
    cl_mem device_input2;  // Input array on the device
    cl_mem device_output;
    

    // Use clCreateBuffer() to create a buffer object (d_A) 
    // that will contain the data from the host array A
    std::cout << "Creating buffer...\n" << std::flush; 
    device_input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    device_input2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, NULL);

    std::cout << "Load data to device...\n" << std::flush; 
    // Use clEnqueueWriteBuffer() to write input array A to
    // the device buffer bufferA
    cl_event evt1, evt2, evt3, evt4;

    status = clEnqueueWriteBuffer(
        cmdQueue, 
        device_input1, 
        CL_FALSE, 
        0, 
        buffer_size,
        host_input,
        0,
        NULL,
        &evt1);

    status = clEnqueueWriteBuffer(
        cmdQueue, 
        device_input2, 
        CL_FALSE, 
        0, 
        buffer_size,
        host_input,
        1,
        &evt1,
        &evt2);

    device_output = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);

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

    std::cout << "program == " << (program == NULL) << "\n";

    cl_kernel kernel1 = NULL;
    cl_kernel kernel2 = NULL;

    std::cout << "Create kernel1...\n" << std::flush; 
    // Use clCreateKernel() to create a kernel1 from the 
    // vector addition function (named "vecadd")
    kernel1 = clCreateKernel(program, NAME1, &status);
    std::cout << "kernel1 == " << (kernel1 == NULL) << "\n";

    std::cout << "Create kernel2...\n" << std::flush; 
    // Use clCreateKernel() to create a kernel1 from the 
    // vector addition function (named "vecadd")
    kernel2 = clCreateKernel(program, NAME2, &status);
    std::cout << "kernel1 == " << (kernel1 == NULL) << "\n";

    std::cout << "Set arguments...\n" << std::flush; 
    
    // Associate the input and output buffers with the 
    // kernel1 
    // using clSetKernelArg()
    status  = clSetKernelArg(
        kernel1, 
        0, 
        sizeof(cl_mem), 
        &device_input1);
    CHECK_CL(status, "Can't assign args0");
    status |= clSetKernelArg(
        kernel1, 
        1, 
        sizeof(cl_mem), 
        &device_input2);
    CHECK_CL(status, "Can't assign args1");
    status |= clSetKernelArg(
        kernel1, 
        2, 
        sizeof(cl_mem), 
        &device_output);
    CHECK_CL(status, "Can't assign args2");

    std::cout << "Set arguments...\n" << std::flush; 

    status  = clSetKernelArg(
        kernel2, 
        0, 
        sizeof(cl_mem), 
        &device_input1);
    CHECK_CL(status, "Can't assign args0");
    status |= clSetKernelArg(
        kernel2, 
        1, 
        sizeof(cl_mem), 
        &device_input2);
    CHECK_CL(status, "Can't assign args1");
    status |= clSetKernelArg(
        kernel2, 
        2, 
        sizeof(cl_mem), 
        &device_output);
    CHECK_CL(status, "Can't assign args2");

    size_t maxWorkGroupSize = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    cl_event events[2] = {evt1, evt2};
    clWaitForEvents(2, events);
    size_t global_size[1] =  {buffer_size / sizeof(float)};
    // maxWorkGroupSize = 1;

    // size_t localWorkSize[2];
    // localWorkSize[0] = 8;
    // localWorkSize[1] = 8;

    std::cout << "Launch kernel1...\n" << std::flush; 
    status = clEnqueueNDRangeKernel(
        cmdQueue, 
        kernel1, 
        1, 
        NULL, 
        global_size, 
        &maxWorkGroupSize, 
        0, 
        NULL, 
        &evt1);

    CHECK_CL(status, "Can't enqueue kernel1");

    std::cout << "Launch kernel2...\n" << std::flush; 
    status = clEnqueueNDRangeKernel(
        cmdQueue, 
        kernel2, 
        1, 
        NULL, 
        global_size, 
        &maxWorkGroupSize, 
        1, 
        &evt1, 
        &evt2);

    CHECK_CL(status, "Can't enqueue kernel2");

    

    std::cout << "Fetch result...\n" << std::flush; 
    clEnqueueReadBuffer(
        cmdQueue, 
        device_output, 
        CL_TRUE, 
        0,
        buffer_size, 
        host_output, 
        1, 
        &evt2, 
        NULL);

    std::cout << "Check correctness...\n" << std::flush; 
    for (int i = 0; i < buffer_size / (sizeof(float)); ++i) {
        float test_data = host_input[i] + host_input[i];
        test_data = test_data * host_input[i] - host_input[i];
        if (std::abs(test_data - host_output[i]) / host_output[i] > 1e-5) {
            std::cout << "Error: " << test_data << " vs. " << host_output[i] << "\n";
        }
    }

    std::cout << "Clean up...\n" << std::flush; 
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(device_input1);
    clReleaseMemObject(device_input2);
    clReleaseMemObject(device_output);
    clReleaseContext(context);

    // Free host resources
    free(host_input);
    free(host_output);
    free(kernel_source);

    std::cout << "Done!\n" << std::flush; 
    return 0;
}