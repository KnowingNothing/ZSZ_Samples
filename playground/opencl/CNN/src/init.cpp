#include "cnn.h"

using namespace std;

void CNN::init() {
  //初始化数据
  int len1 =
      width_image_input_CNN * height_image_input_CNN * num_patterns_train_CNN;
  data_input_train = new float[len1];
  init_variable(data_input_train, -1.0, len1);

  int len2 = num_map_output_CNN * num_patterns_train_CNN;
  data_output_train = new float[len2];
  init_variable(data_output_train, -0.8, len2);

  int len3 =
      width_image_input_CNN * height_image_input_CNN * num_patterns_test_CNN;
  data_input_test = new float[len3];
  init_variable(data_input_test, -1.0, len3);

  int len4 = num_map_output_CNN * num_patterns_test_CNN;
  data_output_test = new float[len4];
  init_variable(data_output_test, -0.8, len4);

  std::fill(E_weight_C1, E_weight_C1 + len_weight_C1_CNN, 0.0);
  std::fill(E_bias_C1, E_bias_C1 + len_bias_C1_CNN, 0.0);
  std::fill(E_weight_S2, E_weight_S2 + len_weight_S2_CNN, 0.0);
  std::fill(E_bias_S2, E_bias_S2 + len_bias_S2_CNN, 0.0);
  std::fill(E_weight_C3, E_weight_C3 + len_weight_C3_CNN, 0.0);
  std::fill(E_bias_C3, E_bias_C3 + len_bias_C3_CNN, 0.0);
  std::fill(E_weight_S4, E_weight_S4 + len_weight_S4_CNN, 0.0);
  std::fill(E_bias_S4, E_bias_S4 + len_bias_S4_CNN, 0.0);
  E_weight_C5 = new float[len_weight_C5_CNN];
  std::fill(E_weight_C5, E_weight_C5 + len_weight_C5_CNN, 0.0);
  E_bias_C5 = new float[len_bias_C5_CNN];
  std::fill(E_bias_C5, E_bias_C5 + len_bias_C5_CNN, 0.0);
  E_weight_output = new float[len_weight_output_CNN];
  std::fill(E_weight_output, E_weight_output + len_weight_output_CNN, 0.0);
  E_bias_output = new float[len_bias_output_CNN];
  std::fill(E_bias_output, E_bias_output + len_bias_output_CNN, 0.0);

  //初始化Weight
  initWeightThreshold();
  //读取MNIST数据
  getSrcData();

#ifdef TARGET_GPU
  std::cout << "Getting device...\n" << std::flush;
  cl_uint num_platforms;
  CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms), "Can't get platform ids");
  cl_platform_id platforms[num_platforms];
  CHECK_CL(clGetPlatformIDs(num_platforms, platforms, NULL),
           "Can't get platforms");
  if (num_platforms <= platform_id) {
    std::cerr << "Can't get expected platform: " << platform_id << " within "
              << num_platforms << " platforms.\n";
    abort();
  }

  this->platform = platforms[platform_id];
  cl_uint num_devices;
  CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices),
           "Can't get device number.");
  cl_device_id devices[num_devices];
  CHECK_CL(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL),
      "Can't get device ids.");
  if (num_devices <= device_id) {
    std::cerr << "Can't get expected device: " << device_id << " within "
              << num_devices << " devices.\n";
    abort();
  }

  this->device = devices[device_id];

  std::cout << "Loading kernel source...\n" << std::flush;
  this->forward_source[0] = load_kernel_source(this->forward_source_C1_name);
  this->forward_source[1] = load_kernel_source(this->forward_source_S2_name);
  this->forward_source[2] = load_kernel_source(this->forward_source_C3_name);
  assert(this->forward_source[0] != NULL);
  std::cout << this->forward_source[2];
  this->forward_source[3] = load_kernel_source(this->forward_source_S4_name);
  this->forward_source[4] = load_kernel_source(this->forward_source_C5_name);
  this->forward_source[5] =
      load_kernel_source(this->forward_source_output_name);

  this->backward_source[0] =
      load_kernel_source(this->backward_source_input_weight_name);
  this->backward_source[1] =
      load_kernel_source(this->backward_source_C1_weight_name);
  this->backward_source[2] =
      load_kernel_source(this->backward_source_S2_weight_name);
  this->backward_source[3] =
      load_kernel_source(this->backward_source_C3_weight_name);
  this->backward_source[4] =
      load_kernel_source(this->backward_source_S4_weight_name);
  this->backward_source[5] =
      load_kernel_source(this->backward_source_C1_input_name);
  this->backward_source[6] =
      load_kernel_source(this->backward_source_S2_input_name);
  this->backward_source[7] =
      load_kernel_source(this->backward_source_C3_input_name);
  this->backward_source[8] =
      load_kernel_source(this->backward_source_S4_input_name);
  this->backward_source[9] = load_kernel_source(this->backward_source_C5_name);
  this->backward_source[10] =
      load_kernel_source(this->backward_source_output_name);

  this->update_source[0] = load_kernel_source(this->update_source_name);

  cl_int status;
  this->context =
      clCreateContext(NULL, num_devices, devices, NULL, NULL, &status);

  cl_command_queue_properties properties[]{CL_QUEUE_PROPERTIES,
                                           CL_QUEUE_PROFILING_ENABLE, 0};
  this->cmdQueue = clCreateCommandQueueWithProperties(
      this->context, this->device, properties, &status);

  std::cout << "Creating buffers...\n" << std::flush;
  /*
   * data set
   */
  this->device_data_input_train = clCreateBuffer(
      this->context, CL_MEM_READ_ONLY, len1 * sizeof(float), NULL, &status);
  this->device_data_output_train = clCreateBuffer(
      this->context, CL_MEM_READ_ONLY, len2 * sizeof(float), NULL, &status);
  this->device_data_input_test = clCreateBuffer(
      this->context, CL_MEM_READ_ONLY, len3 * sizeof(float), NULL, &status);
  this->device_data_output_test = clCreateBuffer(
      this->context, CL_MEM_READ_ONLY, len4 * sizeof(float), NULL, &status);

  this->device_tbl = clCreateBuffer(this->context, CL_MEM_READ_ONLY,
                                    6 * 16 * sizeof(bool), NULL, &status);

  /*
   * weigth and bias
   */
  this->device_weight_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C1_CNN * sizeof(float), NULL, &status);
  this->device_bias_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C1_CNN * sizeof(float), NULL, &status);
  this->device_weight_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_S2_CNN * sizeof(float), NULL, &status);
  this->device_bias_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_S2_CNN * sizeof(float), NULL, &status);
  this->device_weight_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C3_CNN * sizeof(float), NULL, &status);
  this->device_bias_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C3_CNN * sizeof(float), NULL, &status);
  this->device_weight_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_S4_CNN * sizeof(float), NULL, &status);
  this->device_bias_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_S4_CNN * sizeof(float), NULL, &status);
  this->device_weight_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C5_CNN * sizeof(float), NULL, &status);
  this->device_bias_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C5_CNN * sizeof(float), NULL, &status);
  this->device_weight_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_output_CNN * sizeof(float), NULL, &status);
  this->device_bias_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_output_CNN * sizeof(float), NULL, &status);

  /*
   * accum grad of weight and bias
   */
  this->device_E_weight_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C1_CNN * sizeof(float), NULL, &status);
  this->device_E_bias_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C1_CNN * sizeof(float), NULL, &status);
  this->device_E_weight_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_S2_CNN * sizeof(float), NULL, &status);
  this->device_E_bias_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_S2_CNN * sizeof(float), NULL, &status);
  this->device_E_weight_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C3_CNN * sizeof(float), NULL, &status);
  this->device_E_bias_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C3_CNN * sizeof(float), NULL, &status);
  this->device_E_weight_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_S4_CNN * sizeof(float), NULL, &status);
  this->device_E_bias_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_S4_CNN * sizeof(float), NULL, &status);
  this->device_E_weight_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C5_CNN * sizeof(float), NULL, &status);
  this->device_E_bias_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C5_CNN * sizeof(float), NULL, &status);
  this->device_E_weight_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_output_CNN * sizeof(float), NULL, &status);
  this->device_E_bias_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_output_CNN * sizeof(float), NULL, &status);

  /*
   * output data
   */
  // this->device_neuron_input; // data_single_image
  this->device_neuron_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_C1_CNN * sizeof(float), NULL, &status);
  this->device_neuron_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_S2_CNN * sizeof(float), NULL, &status);
  this->device_neuron_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_C3_CNN * sizeof(float), NULL, &status);
  this->device_neuron_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_S4_CNN * sizeof(float), NULL, &status);
  this->device_neuron_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_C5_CNN * sizeof(float), NULL, &status);
  this->device_neuron_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_output_CNN * sizeof(float), NULL, &status);

  /*
   * grad of output
   */
  this->device_delta_neuron_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_output_CNN * sizeof(float), NULL, &status);
  this->device_delta_neuron_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_C5_CNN * sizeof(float), NULL, &status);
  this->device_delta_neuron_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_S4_CNN * sizeof(float), NULL, &status);
  this->device_delta_neuron_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_C3_CNN * sizeof(float), NULL, &status);
  this->device_delta_neuron_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_S2_CNN * sizeof(float), NULL, &status);
  this->device_delta_neuron_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_C1_CNN * sizeof(float), NULL, &status);
  this->device_delta_neuron_input =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     num_neuron_input_CNN * sizeof(float), NULL, &status);

  /*
   * grad of weights and bias
   */
  this->device_delta_weight_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C1_CNN * sizeof(float), NULL, &status);
  this->device_delta_bias_C1 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C1_CNN * sizeof(float), NULL, &status);
  this->device_delta_weight_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_S2_CNN * sizeof(float), NULL, &status);
  this->device_delta_bias_S2 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_S2_CNN * sizeof(float), NULL, &status);
  this->device_delta_weight_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C3_CNN * sizeof(float), NULL, &status);
  this->device_delta_bias_C3 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C3_CNN * sizeof(float), NULL, &status);
  this->device_delta_weight_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_S4_CNN * sizeof(float), NULL, &status);
  this->device_delta_bias_S4 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_S4_CNN * sizeof(float), NULL, &status);
  this->device_delta_weight_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_C5_CNN * sizeof(float), NULL, &status);
  this->device_delta_bias_C5 =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_C5_CNN * sizeof(float), NULL, &status);
  this->device_delta_weight_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_weight_output_CNN * sizeof(float), NULL, &status);
  this->device_delta_bias_output =
      clCreateBuffer(this->context, CL_MEM_READ_WRITE,
                     len_bias_output_CNN * sizeof(float), NULL, &status);

  std::cout << "Load data to device...\n" << std::flush;
  // Use clEnqueueWriteBuffer() to write input array A to
  // the device buffer bufferA
  status = clEnqueueWriteBuffer(this->cmdQueue, this->device_data_input_train,
                                CL_TRUE, 0, len1 * sizeof(float),
                                this->data_input_train, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(this->cmdQueue, this->device_data_output_train,
                                CL_TRUE, 0, len2 * sizeof(float),
                                this->data_output_train, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(this->cmdQueue, this->device_data_input_test,
                                CL_TRUE, 0, len3 * sizeof(float),
                                this->data_input_test, 0, NULL, NULL);
  status = clEnqueueWriteBuffer(this->cmdQueue, this->device_data_output_test,
                                CL_TRUE, 0, len4 * sizeof(float),
                                this->data_output_test, 0, NULL, NULL);

  status = clEnqueueWriteBuffer(this->cmdQueue, this->device_tbl, CL_TRUE, 0,
                                6 * 16 * sizeof(bool), this->host_tbl, 0, NULL,
                                NULL);

#define ENQUE_BUFFER(NAME, LEN)                                                \
  status = clEnqueueWriteBuffer(this->cmdQueue, this->device_##NAME, CL_TRUE,  \
                                0, LEN * sizeof(float), NAME, 0, NULL, NULL);

  ENQUE_BUFFER(weight_C1, len_weight_C1_CNN);
  ENQUE_BUFFER(bias_C1, len_bias_C1_CNN);
  ENQUE_BUFFER(weight_S2, len_weight_S2_CNN);
  ENQUE_BUFFER(bias_S2, len_bias_S2_CNN);
  ENQUE_BUFFER(weight_C3, len_weight_C3_CNN);
  ENQUE_BUFFER(bias_C3, len_bias_C3_CNN);
  ENQUE_BUFFER(weight_S4, len_weight_S4_CNN);
  ENQUE_BUFFER(bias_S4, len_bias_S2_CNN);
  ENQUE_BUFFER(weight_C5, len_weight_C5_CNN);
  ENQUE_BUFFER(bias_C5, len_bias_C5_CNN);
  ENQUE_BUFFER(weight_output, len_weight_output_CNN);
  ENQUE_BUFFER(bias_output, len_bias_output_CNN);

#undef ENQUE_BUFFER

  image_offset = 0;
  label_offset = 0;
#define BUILD_FORWARD_PROGRAM(NAME, ID)                                        \
  {                                                                            \
    this->forward_program_##NAME = clCreateProgramWithSource(                  \
        this->context, 1, (const char **)&(forward_source[ID]), NULL,          \
        &status);                                                              \
                                                                               \
    CHECK_CL(status, "create forward program " + std::string(#NAME));          \
    status = clBuildProgram(this->forward_program_##NAME, num_devices,         \
                            devices, NULL, NULL, NULL);                        \
    char *log = new char[1024];                                                \
    clGetProgramBuildInfo(this->forward_program_##NAME, device,                \
                          CL_PROGRAM_BUILD_LOG, 1024, log, NULL);              \
    std::cout << log;                                                          \
    delete[] log;                                                              \
    CHECK_CL(status, "build forward program " + std::string(#NAME));           \
  }

#define BUILD_BACKWARD_PROGRAM(NAME, ID)                                       \
  {                                                                            \
    this->backward_program_##NAME = clCreateProgramWithSource(                 \
        this->context, 1, (const char **)&(backward_source[ID]), NULL,         \
        &status);                                                              \
                                                                               \
    CHECK_CL(status, "create backward program " + std::string(#NAME));         \
    status = clBuildProgram(this->backward_program_##NAME, num_devices,        \
                            devices, NULL, NULL, NULL);                        \
    char *log = new char[1024];                                                \
    clGetProgramBuildInfo(this->backward_program_##NAME, device,               \
                          CL_PROGRAM_BUILD_LOG, 1024, log, NULL);              \
    std::cout << log;                                                          \
    delete[] log;                                                              \
    CHECK_CL(status, "build backward program " + std::string(#NAME));          \
  }

  std::cout << "Build forward program...\n" << std::flush;
  BUILD_FORWARD_PROGRAM(C1, 0);
  BUILD_FORWARD_PROGRAM(S2, 1);
  BUILD_FORWARD_PROGRAM(C3, 2);
  BUILD_FORWARD_PROGRAM(S4, 3);
  BUILD_FORWARD_PROGRAM(C5, 4);
  BUILD_FORWARD_PROGRAM(output, 5);

  BUILD_BACKWARD_PROGRAM(input_weight, 0);
  BUILD_BACKWARD_PROGRAM(C1_weight, 1);
  BUILD_BACKWARD_PROGRAM(S2_weight, 2);
  BUILD_BACKWARD_PROGRAM(C3_weight, 3);
  BUILD_BACKWARD_PROGRAM(S4_weight, 4);
  BUILD_BACKWARD_PROGRAM(C1_input, 5);
  BUILD_BACKWARD_PROGRAM(S2_input, 6);
  BUILD_BACKWARD_PROGRAM(C3_input, 7);
  BUILD_BACKWARD_PROGRAM(S4_input, 8);
  BUILD_BACKWARD_PROGRAM(C5, 9);
  BUILD_BACKWARD_PROGRAM(output, 10);

#undef BUILD_FORWARD_PROGRAM
#undef BUILD_BACKWRD_PROGRAM

  std::cout << "Build update program...\n" << std::flush;
  this->update_program = clCreateProgramWithSource(
      this->context, 1, (const char **)&(this->update_source[0]), NULL,
      &status);
  CHECK_CL(status, "create update program");
  // Build (compile) the program for the devices with
  // clBuildProgram()
  status = clBuildProgram(this->update_program, num_devices, devices, NULL,
                          NULL, NULL);
  CHECK_CL(status, "build update program");

  std::cout << "Create kernel...\n" << std::flush;
  // Use clCreateKernel() to create a kernel from the
  // vector addition function (named "vecadd")
  this->forward_C1_kernel =
      clCreateKernel(this->forward_program_C1, "kernel_forward_C1", &status);
  this->forward_S2_kernel =
      clCreateKernel(this->forward_program_S2, "kernel_forward_S2", &status);
  this->forward_C3_kernel =
      clCreateKernel(this->forward_program_C3, "kernel_forward_C3", &status);
  this->forward_S4_kernel =
      clCreateKernel(this->forward_program_S4, "kernel_forward_S4", &status);
  this->forward_C5_kernel =
      clCreateKernel(this->forward_program_C5, "kernel_forward_C5", &status);
  this->forward_output_kernel = clCreateKernel(
      this->forward_program_output, "kernel_forward_output", &status);

  this->backward_C1_weight_kernel = clCreateKernel(
      this->backward_program_C1_weight, "kernel_backward_C1_weight", &status);
  this->backward_S2_weight_kernel = clCreateKernel(
      this->backward_program_S2_weight, "kernel_backward_S2_weight", &status);
  this->backward_C3_weight_kernel = clCreateKernel(
      this->backward_program_C3_weight, "kernel_backward_C3_weight", &status);
  this->backward_S4_weight_kernel = clCreateKernel(
      this->backward_program_S4_weight, "kernel_backward_S4_weight", &status);
  this->backward_C5_kernel =
      clCreateKernel(this->backward_program_C5, "kernel_backward_C5", &status);
  this->backward_input_weight_kernel =
      clCreateKernel(this->backward_program_input_weight,
                     "kernel_backward_input_weight", &status);

  this->backward_C1_input_kernel = clCreateKernel(
      this->backward_program_C1_input, "kernel_backward_C1_input", &status);
  this->backward_S2_input_kernel = clCreateKernel(
      this->backward_program_S2_input, "kernel_backward_S2_input", &status);
  this->backward_C3_input_kernel = clCreateKernel(
      this->backward_program_C3_input, "kernel_backward_C3_input", &status);
  this->backward_S4_input_kernel = clCreateKernel(
      this->backward_program_S4_input, "kernel_backward_S4_input", &status);
  this->backward_output_kernel = clCreateKernel(
      this->backward_program_output, "kernel_backward_output", &status);

  this->update_kernel =
      clCreateKernel(this->update_program, "kernel_update", &status);

#define X(xx) (assert((xx) != NULL));
  X(this->forward_C1_kernel)
  X(this->forward_S2_kernel)
  X(this->forward_C3_kernel)
  X(this->forward_S4_kernel)
  X(this->forward_C5_kernel)
  X(this->forward_output_kernel)
  X(this->backward_input_weight_kernel)
  X(this->backward_C1_weight_kernel)
  X(this->backward_S2_weight_kernel)
  X(this->backward_C3_weight_kernel)
  X(this->backward_S4_weight_kernel)
  X(this->backward_C5_kernel)
  X(this->backward_output_kernel)
  X(this->backward_C1_input_kernel)
  X(this->backward_S2_input_kernel)
  X(this->backward_C3_input_kernel)
  X(this->backward_S4_input_kernel)
  X(this->update_kernel)
#undef X
#endif
}

float CNN::uniform_rand(float min, float max) {
  // std::mt19937 gen(1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dst(min, max);
  return dst(gen);
}

bool CNN::uniform_rand(float *src, int len, float min, float max) {
  for (int i = 0; i < len; i++) {
    src[i] = uniform_rand(min, max);
  }
  return true;
}

bool CNN::initWeightThreshold() {
  srand(time(0) + rand());
  const float scale = 6.0;

  float min_ = -std::sqrt(scale / (25.0 + 150.0));
  float max_ = std::sqrt(scale / (25.0 + 150.0));
  uniform_rand(weight_C1, len_weight_C1_CNN, min_, max_);
  for (int i = 0; i < len_bias_C1_CNN; i++) {
    bias_C1[i] = 0.0;
  }

  min_ = -std::sqrt(scale / (4.0 + 1.0));
  max_ = std::sqrt(scale / (4.0 + 1.0));
  uniform_rand(weight_S2, len_weight_S2_CNN, min_, max_);
  for (int i = 0; i < len_bias_S2_CNN; i++) {
    bias_S2[i] = 0.0;
  }

  min_ = -std::sqrt(scale / (150.0 + 400.0));
  max_ = std::sqrt(scale / (150.0 + 400.0));
  uniform_rand(weight_C3, len_weight_C3_CNN, min_, max_);
  for (int i = 0; i < len_bias_C3_CNN; i++) {
    bias_C3[i] = 0.0;
  }

  min_ = -std::sqrt(scale / (4.0 + 1.0));
  max_ = std::sqrt(scale / (4.0 + 1.0));
  uniform_rand(weight_S4, len_weight_S4_CNN, min_, max_);
  for (int i = 0; i < len_bias_S4_CNN; i++) {
    bias_S4[i] = 0.0;
  }

  min_ = -std::sqrt(scale / (400.0 + 3000.0));
  max_ = std::sqrt(scale / (400.0 + 3000.0));
  uniform_rand(weight_C5, len_weight_C5_CNN, min_, max_);
  for (int i = 0; i < len_bias_C5_CNN; i++) {
    bias_C5[i] = 0.0;
  }

  min_ = -std::sqrt(scale / (120.0 + 10.0));
  max_ = std::sqrt(scale / (120.0 + 10.0));
  uniform_rand(weight_output, len_weight_output_CNN, min_, max_);
  for (int i = 0; i < len_bias_output_CNN; i++) {
    bias_output[i] = 0.0;
  }

  return true;
}

#ifdef TARGET_GPU
char *CNN::load_kernel_source(std::string filename) {
  std::ifstream kernel_src(filename, std::ifstream::in);
  if (!kernel_src.good()) {
    std::cerr << "Can't open source file: " << filename << "\n";
    abort();
  }
  char *kernel_source = NULL;
  kernel_src.seekg(0, kernel_src.end);
  int kernel_size = kernel_src.tellg();
  kernel_source = (char *)malloc(kernel_size + 1);
  kernel_src.seekg(0, kernel_src.beg);
  kernel_src.read(kernel_source, kernel_size);
  kernel_source[kernel_size] = '\0';

  kernel_src.close();
  assert(kernel_source != NULL);
  return kernel_source;
}
#endif
