/*
 * predict.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;

int CNN::predict(const unsigned char *data, int width, int height) {
  assert(data && width == width_image_input_CNN &&
         height == height_image_input_CNN);

  const float scale_min = -1;
  const float scale_max = 1;

  float tmp[width_image_input_CNN * height_image_input_CNN];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      tmp[y * width + x] =
          (data[y * width + x] / 255.0) * (scale_max - scale_min) + scale_min;
      // std::cout << tmp[y * width + x] << " ";
      // printf("%.f ", tmp[y * width + x]);
    }
    // std::cout << std::endl;
  }
  // std::cout << std::endl;

  data_single_image = &tmp[0];
#ifdef TARGET_GPU
  cl_int status;
  this->device_data_pointer = clCreateBuffer(
      this->context, CL_MEM_READ_ONLY,
      width_image_input_CNN * height_image_input_CNN * sizeof(float), NULL,
      &status);
  CHECK_CL(status, "");
  status = clEnqueueWriteBuffer(
      this->cmdQueue, this->device_data_pointer, CL_TRUE, 0,
      width_image_input_CNN * height_image_input_CNN * sizeof(float), tmp, 0,
      NULL, NULL);
  CHECK_CL(status, "");
#endif
  Forward_C1();
#ifdef TARGET_GPU
//   status = clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_C1,
//                                CL_TRUE, 0, num_neuron_C1_CNN * sizeof(float),
//                                this->neuron_C1, 0, NULL, NULL);
//   for (int i = 0; i < num_neuron_C1_CNN; ++i) {
// 	  if (neuron_C1[i] != 0) {
// 		  std::cout << neuron_C1[i] << " ";
// 	  }
//   }
//   std::cout << "\n";
#endif
  Forward_S2();
#ifdef TARGET_GPU
//   status = clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_S2,
//                                CL_TRUE, 0, num_neuron_S2_CNN * sizeof(float),
//                                this->neuron_S2, 0, NULL, NULL);
//   for (int i = 0; i < num_neuron_S2_CNN; ++i) {
// 	  if (neuron_S2[i] != 0) {
// 		  std::cout << neuron_S2[i] << " ";
// 	  }
//   }
//   std::cout << "\n";
#endif
  Forward_C3();
#ifdef TARGET_GPU
//   status = clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_C3,
//                                CL_TRUE, 0, num_neuron_C3_CNN * sizeof(float),
//                                this->neuron_C3, 0, NULL, NULL);
//   for (int i = 0; i < num_neuron_C3_CNN; ++i) {
// 	  if (neuron_C3[i] != 0) {
// 		  std::cout << neuron_C3[i] << " ";
// 	  }
//   }
//   std::cout << "\n";
#endif
  Forward_S4();
#ifdef TARGET_GPU
//   status = clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_S4,
//                                CL_TRUE, 0, num_neuron_S4_CNN * sizeof(float),
//                                this->neuron_S4, 0, NULL, NULL);
//   for (int i = 0; i < num_neuron_S4_CNN; ++i) {
// 	  if (neuron_S4[i] != 0) {
// 		  std::cout << neuron_S4[i] << " ";
// 	  }
//   }
//   std::cout << "\n";
#endif
  Forward_C5();
#ifdef TARGET_GPU
//   status = clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_C5,
//                                CL_TRUE, 0, num_neuron_C5_CNN * sizeof(float),
//                                this->neuron_C5, 0, NULL, NULL);
//   for (int i = 0; i < num_neuron_C5_CNN; ++i) {
// 	  if (neuron_C5[i] != 0) {
// 		  std::cout << neuron_C5[i] << " ";
// 	  }
//   }
//   std::cout << "\n";
#endif
  Forward_output();

#ifdef TARGET_GPU
  status = clEnqueueReadBuffer(this->cmdQueue, this->device_neuron_output,
                               CL_TRUE, 0, num_neuron_output_CNN * sizeof(float),
                               this->neuron_output, 0, NULL, NULL);
#endif

  int pos = -1;
  float max_value = -9999.0;

  for (int i = 0; i < num_neuron_output_CNN; i++) {
    std::cout << " " << neuron_output[i] << " " << std::endl;
    if (neuron_output[i] > max_value) {
      max_value = neuron_output[i];
      pos = i;
    }
  }

  return pos;
}
