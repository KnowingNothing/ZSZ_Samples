/*
 * predict.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;

int CNN::predict(const unsigned char* data, int width, int height)
{
	assert(data && width == width_image_input_CNN && height == height_image_input_CNN);

	const float scale_min = -1;
	const float scale_max = 1;

	float tmp[width_image_input_CNN * height_image_input_CNN];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			tmp[y * width + x] = (data[y * width + x] / 255.0) * (scale_max - scale_min) + scale_min;
			//std::cout << tmp[y * width + x] << " ";
			//printf("%.f ", tmp[y * width + x]);
		}
		//std::cout << std::endl;
	}
	//std::cout << std::endl;

	data_single_image = &tmp[0];
	Forward_C1();
	Forward_S2();
	Forward_C3();
	Forward_S4();
	Forward_C5();
	Forward_output();

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


