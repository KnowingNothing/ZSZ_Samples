#include "cnn.h"

using namespace std;

#ifdef FORWARD_GPU
std::string status2str(cl_int status) {
  if (status == CL_SUCCESS) {
    return "CL_SUCCESS";
  } else if (status == CL_INVALID_VALUE) {
    return "CL_​INVALID_​VALUE";
  } else if (status == CL_OUT_OF_HOST_MEMORY) {
    return "CL_​OUT_​OF_​HOST_​MEMORY";
  } else {
    return "Unknown status: " + std::to_string(status);
  }
}

std::string get_device_info(cl_device_id device, cl_device_info params) {
  size_t buffer_size;
  CHECK_CL(clGetDeviceInfo(device, params, 0, NULL, &buffer_size),
           "Can't get device info buffer size.");
  char *buffer = (char *)malloc(buffer_size);
  CHECK_CL(clGetDeviceInfo(device, params, buffer_size, buffer, NULL),
           "Can't get device info.");
  std::string ret = std::string(buffer);
  if (params == CL_DEVICE_MAX_COMPUTE_UNITS ||
      params == CL_DEVICE_MAX_CLOCK_FREQUENCY ||
      params == CL_DEVICE_DOUBLE_FP_CONFIG) {
    cl_uint num = *((cl_uint *)buffer);
    std::ostringstream oss;
    oss << num;
    ret = oss.str();
  } else if (params == CL_DEVICE_GLOBAL_MEM_SIZE) {
    cl_ulong num = *((cl_ulong *)buffer);
    num /= 1e9;
    std::ostringstream oss;
    oss << num;
    ret = oss.str();
  }
  free(buffer);
  return ret;
}
#endif

#ifdef FORWARD_GPU
CNN::CNN(int platform_id, int device_id)
#else
CNN::CNN()
#endif
{
  data_input_train = NULL;
  data_output_train = NULL;
  data_input_test = NULL;
  data_output_test = NULL;
  data_single_image = NULL;
  data_single_label = NULL;
  E_weight_C5 = NULL;
  E_bias_C5 = NULL;
  E_weight_output = NULL;
  E_bias_output = NULL;
#ifdef FORWARD_GPU
  this->platform_id = platform_id;
  this->device_id = device_id;
#endif
  std::cout << "Create" << std::endl;
}

CNN::~CNN() { release(); }

void CNN::release() {
  if (data_input_train) {
    delete[] data_input_train;
    data_input_train = NULL;
  }
  if (data_output_train) {
    delete[] data_output_train;
    data_output_train = NULL;
  }
  if (data_input_test) {
    delete[] data_input_test;
    data_input_test = NULL;
  }
  if (data_output_test) {
    delete[] data_output_test;
    data_output_test = NULL;
  }

  if (E_weight_C5) {
    delete[] E_weight_C5;
    E_weight_C5 = NULL;
  }
  if (E_bias_C5) {
    delete[] E_bias_C5;
    E_bias_C5 = NULL;
  }
  if (E_weight_output) {
    delete[] E_weight_output;
    E_weight_output = NULL;
  }
  if (E_bias_output) {
    delete[] E_bias_output;
    E_bias_output = NULL;
  }

#ifdef FORWARD_GPU
  for (int i = 0; i < 6; ++i) {
    if (forward_source[i]) {
      delete[] forward_source[i];
    }
  }
  for (int i = 0; i < 11; ++i) {
    if (backward_source[i]) {
      delete[] backward_source[i];
    }
  }
  for (int i = 0; i < 1; ++i) {
    if (update_source[i]) {
      delete[] update_source[i];
    }
  }

  clReleaseKernel(forward_C1_kernel);
  clReleaseKernel(forward_S2_kernel);
  clReleaseKernel(forward_C3_kernel);
  clReleaseKernel(forward_S4_kernel);
  clReleaseKernel(forward_C5_kernel);
  clReleaseKernel(forward_output_kernel);
  clReleaseKernel(backward_input_weight_kernel);
  clReleaseKernel(backward_C1_weight_kernel);
  clReleaseKernel(backward_S2_weight_kernel);
  clReleaseKernel(backward_C3_weight_kernel);
  clReleaseKernel(backward_S4_weight_kernel);
  clReleaseKernel(backward_C1_input_kernel);
  clReleaseKernel(backward_S2_input_kernel);
  clReleaseKernel(backward_C3_input_kernel);
  clReleaseKernel(backward_S4_input_kernel);
  clReleaseKernel(backward_C5_kernel);
  clReleaseKernel(backward_output_kernel);
  clReleaseKernel(update_kernel);

  clReleaseProgram(forward_program_C1);
  clReleaseProgram(forward_program_S2);
  clReleaseProgram(forward_program_C3);
  clReleaseProgram(forward_program_S4);
  clReleaseProgram(forward_program_C5);
  clReleaseProgram(forward_program_output);
  clReleaseProgram(backward_program_input_weight);
  clReleaseProgram(backward_program_C1_weight);
  clReleaseProgram(backward_program_S2_weight);
  clReleaseProgram(backward_program_C3_weight);
  clReleaseProgram(backward_program_S4_weight);
  clReleaseProgram(backward_program_C1_input);
  clReleaseProgram(backward_program_S2_input);
  clReleaseProgram(backward_program_C3_input);
  clReleaseProgram(backward_program_S4_input);
  clReleaseProgram(backward_program_C5);
  clReleaseProgram(backward_program_output);
  clReleaseProgram(update_program);

  clReleaseCommandQueue(cmdQueue);
  clReleaseContext(context);

  clReleaseMemObject(
      device_data_input_train); //原始标准输入数据，训练,范围：[-1, 1]
  clReleaseMemObject(
      device_data_output_train); //原始标准期望结果，训练,取值：-0.8/0.8
  clReleaseMemObject(
      device_data_input_test); //原始标准输入数据，测试,范围：[-1, 1]
  clReleaseMemObject(
      device_data_output_test); //原始标准期望结果，测试,取值：-0.8/0.8

  clReleaseMemObject(device_tbl);

  clReleaseMemObject(device_weight_C1);
  clReleaseMemObject(device_bias_C1);
  clReleaseMemObject(device_weight_S2);
  clReleaseMemObject(device_bias_S2);
  clReleaseMemObject(device_weight_C3);
  clReleaseMemObject(device_bias_C3);
  clReleaseMemObject(device_weight_S4);
  clReleaseMemObject(device_bias_S4);
  clReleaseMemObject(device_weight_C5);
  clReleaseMemObject(device_bias_C5);
  clReleaseMemObject(device_weight_output);
  clReleaseMemObject(device_bias_output);

  clReleaseMemObject(device_E_weight_C1); //累积误差
  clReleaseMemObject(device_E_bias_C1);
  clReleaseMemObject(device_E_weight_S2);
  clReleaseMemObject(device_E_bias_S2);
  clReleaseMemObject(device_E_weight_C3);
  clReleaseMemObject(device_E_bias_C3);
  clReleaseMemObject(device_E_weight_S4);
  clReleaseMemObject(device_E_bias_S4);
  clReleaseMemObject(device_E_weight_C5);
  clReleaseMemObject(device_E_bias_C5);
  clReleaseMemObject(device_E_weight_output);
  clReleaseMemObject(device_E_bias_output);

  //   clReleaseMemObject(device_neuron_input); // data_single_image
  clReleaseMemObject(device_neuron_C1);
  clReleaseMemObject(device_neuron_S2);
  clReleaseMemObject(device_neuron_C3);
  clReleaseMemObject(device_neuron_S4);
  clReleaseMemObject(device_neuron_C5);
  clReleaseMemObject(device_neuron_output);

  clReleaseMemObject(device_delta_neuron_output); //神经元误差
  clReleaseMemObject(device_delta_neuron_C5);
  clReleaseMemObject(device_delta_neuron_S4);
  clReleaseMemObject(device_delta_neuron_C3);
  clReleaseMemObject(device_delta_neuron_S2);
  clReleaseMemObject(device_delta_neuron_C1);
  clReleaseMemObject(device_delta_neuron_input);

  clReleaseMemObject(device_delta_weight_C1); //权值、阈值误差
  clReleaseMemObject(device_delta_bias_C1);
  clReleaseMemObject(device_delta_weight_S2);
  clReleaseMemObject(device_delta_bias_S2);
  clReleaseMemObject(device_delta_weight_C3);
  clReleaseMemObject(device_delta_bias_C3);
  clReleaseMemObject(device_delta_weight_S4);
  clReleaseMemObject(device_delta_bias_S4);
  clReleaseMemObject(device_delta_weight_C5);
  clReleaseMemObject(device_delta_bias_C5);
  clReleaseMemObject(device_delta_weight_output);
  clReleaseMemObject(device_delta_bias_output);

#endif
}
