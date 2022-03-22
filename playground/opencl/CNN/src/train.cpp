#include "cnn.h"
#include <chrono>

using namespace std;

#define PROFILE_LAYER

struct timeval tsBegin, tsEnd, ToltsBegin, ToltsEnd;
long t1Duration;

int CNN::get_index(int x, int y, int channel, int width, int height,
                   int depth) {
  assert(x >= 0 && x < width);
  assert(y >= 0 && y < height);
  assert(channel >= 0 && channel < depth);
  return (height * channel + y) * width + x;
}

bool CNN::train() {
  std::cout << "training" << std::endl;
  int iter = 0;
  double profile_layer[2][7] = {{0.0}};
#define PROFILE_FORWARD(N, NAME)                                               \
  {                                                                            \
    t1 = std::chrono::high_resolution_clock::now();                            \
    Forward_##NAME();                                                          \
    t2 = std::chrono::high_resolution_clock::now();                            \
    std::chrono::duration<double> time_span =                                  \
        chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);         \
    profile_layer[0][N] = time_span.count() * 1e3;                             \
  }
#define PROFILE_BACKWARD(N, NAME)                                              \
  {                                                                            \
    t1 = std::chrono::high_resolution_clock::now();                            \
    Backward_##NAME();                                                         \
    t2 = std::chrono::high_resolution_clock::now();                            \
    std::chrono::duration<double> time_span =                                  \
        chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);         \
    profile_layer[1][N] = time_span.count() * 1e3;                             \
  }

#ifdef FORWARD_GPU
  this->device_data_pointer = this->device_data_input_train;
  this->device_label_pointer = this->device_data_output_train;
#endif
  for (iter = 0; iter < num_epochs_CNN; iter++) {
    std::cout << "epoch: " << iter + 1 << std::endl;
    gettimeofday(&ToltsBegin, NULL);
    for (int i = 0; i < num_patterns_train_CNN; i++) {

      if (i % 1000 == 0) {
        gettimeofday(&tsBegin, NULL);
      }
      // 1 输入模式顺传播
      data_single_image = data_input_train + i * num_neuron_input_CNN;
      data_single_label = data_output_train + i * num_neuron_output_CNN;
#ifdef FORWARD_GPU
      image_offset = i * num_neuron_input_CNN;
      label_offset = i * num_neuron_output_CNN;
#endif

      memcpy(neuron_input, data_single_image,
             num_neuron_input_CNN * sizeof(float));

#ifdef PROFILE_LAYER
      std::chrono::high_resolution_clock::time_point t1, t2;
      if (i % 1000 == 0) {
        PROFILE_FORWARD(1, C1)
        PROFILE_FORWARD(2, S2)
        PROFILE_FORWARD(3, C3)
        PROFILE_FORWARD(4, S4)
        PROFILE_FORWARD(5, C5)
        PROFILE_FORWARD(6, output)
      } else {
        Forward_C1();
        Forward_S2();
        Forward_C3();
        Forward_S4();
        Forward_C5();
        Forward_output();
      }
#else
      Forward_C1();
      Forward_S2();
      Forward_C3();
      Forward_S4();
      Forward_C5();
      Forward_output();
#endif

      if (i % 1000 == 0) {
        gettimeofday(&tsEnd, NULL);
        t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) +
                     (tsEnd.tv_usec - tsBegin.tv_usec);
        printf("%dth --> fordward: %1d ms, ", i, t1Duration);
        gettimeofday(&tsBegin, NULL);
      }

// 2 输出误差逆传播
#ifndef BACKWARD_GPU
#ifdef FORWARD_GPU
#define ENQUE_BUFFER(NAME, LEN)                                                \
  clEnqueueReadBuffer(this->cmdQueue, this->device_##NAME, CL_TRUE, 0,         \
                      LEN * sizeof(float), NAME, 0, NULL, NULL);

      ENQUE_BUFFER(neuron_C1, num_neuron_C1_CNN);
      ENQUE_BUFFER(neuron_S2, num_neuron_S2_CNN);
      ENQUE_BUFFER(neuron_C3, num_neuron_C3_CNN);
      ENQUE_BUFFER(neuron_S4, num_neuron_S4_CNN);
      ENQUE_BUFFER(neuron_C5, num_neuron_C5_CNN);
      ENQUE_BUFFER(neuron_output, num_neuron_output_CNN);

#undef ENQUE_BUFFER
#endif
#endif

#ifdef PROFILE_LAYER
      if (i % 1000 == 0) {
        PROFILE_BACKWARD(6, output)
        PROFILE_BACKWARD(5, C5)
        PROFILE_BACKWARD(4, S4)
        PROFILE_BACKWARD(3, C3)
        PROFILE_BACKWARD(2, S2)
        PROFILE_BACKWARD(1, C1)
        PROFILE_BACKWARD(0, input)
      } else {
        Backward_output();
        Backward_C5();
        Backward_S4();
        Backward_C3();
        Backward_S2();
        Backward_C1();
        Backward_input();
      }
#else
      Backward_output();
      Backward_C5();
      Backward_S4();
      Backward_C3();
      Backward_S2();
      Backward_C1();
      Backward_input();

#endif

      if (i % 1000 == 0) {
        gettimeofday(&tsEnd, NULL);
        t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) +
                     (tsEnd.tv_usec - tsBegin.tv_usec);
        printf("backward: %1d ms, ", t1Duration);
        gettimeofday(&tsBegin, NULL);
      }

#ifdef BACKWARD_GPU
#ifndef UPDATE_GPU
#define ENQUE_BUFFER(NAME, LEN)                                                \
  clEnqueueReadBuffer(this->cmdQueue, this->device_##NAME, CL_TRUE, 0,         \
                      LEN * sizeof(float), NAME, 0, NULL, NULL);

      ENQUE_BUFFER(delta_weight_C1, len_weight_C1_CNN);
      ENQUE_BUFFER(delta_bias_C1, len_bias_C1_CNN);
      ENQUE_BUFFER(delta_weight_S2, len_weight_S2_CNN);
      ENQUE_BUFFER(delta_bias_S2, len_bias_S2_CNN);
      ENQUE_BUFFER(delta_weight_C3, len_weight_C3_CNN);
      ENQUE_BUFFER(delta_bias_C3, len_bias_C3_CNN);
      ENQUE_BUFFER(delta_weight_S4, len_weight_S4_CNN);
      ENQUE_BUFFER(delta_bias_S4, len_bias_S2_CNN);
      ENQUE_BUFFER(delta_weight_C5, len_weight_C5_CNN);
      ENQUE_BUFFER(delta_bias_C5, len_bias_C5_CNN);
      ENQUE_BUFFER(delta_weight_output, len_weight_output_CNN);
      ENQUE_BUFFER(delta_bias_output, len_bias_output_CNN);

#undef ENQUE_BUFFER
#endif
#endif

      UpdateWeights();
#ifndef UPDATE_GPU
#ifdef FORWARD_GPU
#define ENQUE_BUFFER(NAME, LEN)                                                \
  clEnqueueWriteBuffer(this->cmdQueue, this->device_##NAME, CL_TRUE, 0,        \
                       LEN * sizeof(float), NAME, 0, NULL, NULL);

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
#endif
#endif

      if (i % 1000 == 0) {
        gettimeofday(&tsEnd, NULL);
        t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) +
                     (tsEnd.tv_usec - tsBegin.tv_usec);
        printf(" UpdateWeights: %1d ms\n", t1Duration);
        printf("\t\tLayer Profile: "
               "C1,S2,C3,S4,C5,output,Binput,BC1,BS2,BC3,BS4,BC5,Boutput:%lf,%"
               "lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
               profile_layer[0][1], profile_layer[0][2], profile_layer[0][3],
               profile_layer[0][4], profile_layer[0][5], profile_layer[0][6],
               profile_layer[1][0], profile_layer[1][1], profile_layer[1][2],
               profile_layer[1][3], profile_layer[1][4], profile_layer[1][5],
               profile_layer[1][6]);
      }
    } // 3 循环记忆训练
      // 4 学习结果判别
    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
    float accuracyRate = 0.0;
    accuracyRate = test();
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << ",test uses " << time_span.count() * 1e3
              << "ms, accuray rate: " << accuracyRate << std::endl;
    if (accuracyRate > accuracy_rate_CNN) {
      saveModelFile("cnn.model");
      std::cout << "generate cnn model" << std::endl;
      break;
    }
    saveModelFile("cnn.model");
    std::cout << "generate cnn model" << std::endl;
    gettimeofday(&ToltsEnd, NULL);
    t1Duration = 1000000L * (ToltsEnd.tv_sec - ToltsBegin.tv_sec) +
                 (ToltsEnd.tv_usec - ToltsBegin.tv_usec);
    printf(" *******  every epoch : %1d s ^_^ \n", t1Duration / 1000000L);
  }

  if (iter == num_epochs_CNN) {
    saveModelFile("cnn.model");
    std::cout << "generate cnn model" << std::endl;
  }
  return true;
}

void CNN::update_weights_bias(const float *delta, float *e_weight,
                              float *weight, int len) {
  for (int i = 0; i < len; i++) {
    e_weight[i] += delta[i] * delta[i];
    weight[i] -=
        learning_rate_CNN * delta[i] / (std::sqrt(e_weight[i]) + eps_CNN);
  }
}

bool CNN::UpdateWeights() {
#ifdef UPDATE_GPU
#define UPDATE(DELTA, E, W, L)                                                 \
  {                                                                            \
    cl_int status;                                                             \
    cl_float lr = learning_rate_CNN;                                           \
    cl_float eps = eps_CNN;                                                    \
    cl_int len = L;                                                            \
    status = clSetKernelArg(this->update_kernel, 0, sizeof(cl_mem), &DELTA);   \
    CHECK_CL(status, "Can't assign args0");                                    \
    status |= clSetKernelArg(this->update_kernel, 1, sizeof(cl_mem), &E);      \
    CHECK_CL(status, "Can't assign args1");                                    \
    status |= clSetKernelArg(this->update_kernel, 2, sizeof(cl_mem), &W);      \
    CHECK_CL(status, "Can't assign args2");                                    \
    status |= clSetKernelArg(this->update_kernel, 3, sizeof(cl_float), &lr);   \
    CHECK_CL(status, "Can't assign args3");                                    \
    status |= clSetKernelArg(this->update_kernel, 4, sizeof(cl_float), &eps);  \
    CHECK_CL(status, "Can't assign args4");                                    \
    status |= clSetKernelArg(this->update_kernel, 5, sizeof(cl_int), &len);    \
    CHECK_CL(status, "Can't assign args5");                                    \
                                                                               \
    cl_event event;                                                            \
    size_t dims[1] = {len};                                                    \
    status = clEnqueueNDRangeKernel(this->cmdQueue, this->update_kernel, 1,    \
                                    NULL, dims, NULL, 0, NULL, &event);        \
    cl_event waits[] = {event};                                                \
    clWaitForEvents(1, waits);                                                 \
    clReleaseEvent(event);                                                     \
  }
#define UPDATE_WEIGHT(NAME)                                                    \
  UPDATE(this->device_delta_weight_##NAME, this->device_E_weight_##NAME,       \
         this->device_weight_##NAME, len_weight_##NAME##_CNN)
#define UPDATE_BIAS(NAME)                                                      \
  UPDATE(this->device_delta_bias_##NAME, this->device_E_bias_##NAME,           \
         this->device_bias_##NAME, len_bias_##NAME##_CNN)
  UPDATE_WEIGHT(C1);
  UPDATE_BIAS(C1);
  UPDATE_WEIGHT(S2);
  UPDATE_BIAS(S2);
  UPDATE_WEIGHT(C3);
  UPDATE_BIAS(C3);
  UPDATE_WEIGHT(S4);
  UPDATE_BIAS(S4);
  UPDATE_WEIGHT(C5);
  UPDATE_BIAS(C5);
  UPDATE_WEIGHT(output);
  UPDATE_BIAS(output);
#undef UPDATE_WEIGHT
#undef UPDATE_BIAS
#undef UPDATE
#else
  update_weights_bias(delta_weight_C1, E_weight_C1, weight_C1,
                      len_weight_C1_CNN);
  update_weights_bias(delta_bias_C1, E_bias_C1, bias_C1, len_bias_C1_CNN);

  update_weights_bias(delta_weight_S2, E_weight_S2, weight_S2,
                      len_weight_S2_CNN);
  update_weights_bias(delta_bias_S2, E_bias_S2, bias_S2, len_bias_S2_CNN);

  update_weights_bias(delta_weight_C3, E_weight_C3, weight_C3,
                      len_weight_C3_CNN);
  update_weights_bias(delta_bias_C3, E_bias_C3, bias_C3, len_bias_C3_CNN);

  update_weights_bias(delta_weight_S4, E_weight_S4, weight_S4,
                      len_weight_S4_CNN);
  update_weights_bias(delta_bias_S4, E_bias_S4, bias_S4, len_bias_S4_CNN);

  update_weights_bias(delta_weight_C5, E_weight_C5, weight_C5,
                      len_weight_C5_CNN);
  update_weights_bias(delta_bias_C5, E_bias_C5, bias_C5, len_bias_C5_CNN);

  update_weights_bias(delta_weight_output, E_weight_output, weight_output,
                      len_weight_output_CNN);
  update_weights_bias(delta_bias_output, E_bias_output, bias_output,
                      len_bias_output_CNN);
#endif
  return true;
}

float CNN::test() {
  int count_accuracy = 0;

#ifdef FORWARD_GPU
  this->device_data_pointer = this->device_data_input_test;
  this->device_label_pointer = this->device_data_output_test;
#endif
  for (int num = 0; num < num_patterns_test_CNN; num++) {
    data_single_image = data_input_test + num * num_neuron_input_CNN;
    data_single_label = data_output_test + num * num_neuron_output_CNN;
#ifdef FORWARD_GPU
    image_offset = num * num_neuron_input_CNN;
    label_offset = num * num_neuron_output_CNN;
#endif

    memcpy(neuron_input, data_single_image,
           num_neuron_input_CNN * sizeof(float));
    Forward_C1(false);
    Forward_S2();
    Forward_C3();
    Forward_S4();
    Forward_C5();
    Forward_output();

#ifdef FORWARD_GPU
    cl_int status;
    status =
        clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_output, CL_TRUE,
                            0, num_neuron_output_CNN * sizeof(float),
                            this->neuron_output, 0, NULL, NULL);
#endif

    int pos_t = -1;
    int pos_y = -2;
    float max_value_t = -9999.0;
    float max_value_y = -9999.0;

    for (int i = 0; i < num_neuron_output_CNN; i++) {
      if (neuron_output[i] > max_value_y) {
        max_value_y = neuron_output[i];
        pos_y = i;
      }

      if (data_single_label[i] > max_value_t) {
        max_value_t = data_single_label[i];
        pos_t = i;
      }
    }

    if (pos_y == pos_t) {
      ++count_accuracy;
    }
    // Copper Sleep(1);
  }
  return (count_accuracy * 1.0 / num_patterns_test_CNN);
}
