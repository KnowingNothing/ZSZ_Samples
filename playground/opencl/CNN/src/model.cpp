/*
 * model.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;

bool CNN::saveModelFile(const char* name)
{
	FILE* fp = fopen(name, "wb");
	if (fp == NULL) {
		return false;
	}

	int width_image_input = width_image_input_CNN;
	int height_image_input = height_image_input_CNN;
	int width_image_C1 = width_image_C1_CNN;
	int height_image_C1 = height_image_C1_CNN;
	int width_image_S2 = width_image_S2_CNN;
	int height_image_S2 = height_image_S2_CNN;
	int width_image_C3 = width_image_C3_CNN;
	int height_image_C3 = height_image_C3_CNN;
	int width_image_S4 = width_image_S4_CNN;
	int height_image_S4 = height_image_S4_CNN;
	int width_image_C5 = width_image_C5_CNN;
	int height_image_C5 = height_image_C5_CNN;
	int width_image_output = width_image_output_CNN;
	int height_image_output = height_image_output_CNN;

	int width_kernel_conv = width_kernel_conv_CNN;
	int height_kernel_conv = height_kernel_conv_CNN;
	int width_kernel_pooling = width_kernel_pooling_CNN;
	int height_kernel_pooling = height_kernel_pooling_CNN;

	int num_map_input = num_map_input_CNN;
	int num_map_C1 = num_map_C1_CNN;
	int num_map_S2 = num_map_S2_CNN;
	int num_map_C3 = num_map_C3_CNN;
	int num_map_S4 = num_map_S4_CNN;
	int num_map_C5 = num_map_C5_CNN;
	int num_map_output = num_map_output_CNN;

	int len_weight_C1 = len_weight_C1_CNN;
	int len_bias_C1 = len_bias_C1_CNN;
	int len_weight_S2 = len_weight_S2_CNN;
	int len_bias_S2 = len_bias_S2_CNN;
	int len_weight_C3 = len_weight_C3_CNN;
	int len_bias_C3 = len_bias_C3_CNN;
	int len_weight_S4 = len_weight_S4_CNN;
	int len_bias_S4 = len_bias_S4_CNN;
	int len_weight_C5 = len_weight_C5_CNN;
	int len_bias_C5 = len_bias_C5_CNN;
	int len_weight_output = len_weight_output_CNN;
	int len_bias_output = len_bias_output_CNN;

	int num_neuron_input = num_neuron_input_CNN;
	int num_neuron_C1 = num_neuron_C1_CNN;
	int num_neuron_S2 = num_neuron_S2_CNN;
	int num_neuron_C3 = num_neuron_C3_CNN;
	int num_neuron_S4 = num_neuron_S4_CNN;
	int num_neuron_C5 = num_neuron_C5_CNN;
	int num_neuron_output = num_neuron_output_CNN;

	fwrite(&width_image_input, sizeof(int), 1, fp);
	fwrite(&height_image_input, sizeof(int), 1, fp);
	fwrite(&width_image_C1, sizeof(int), 1, fp);
	fwrite(&height_image_C1, sizeof(int), 1, fp);
	fwrite(&width_image_S2, sizeof(int), 1, fp);
	fwrite(&height_image_S2, sizeof(int), 1, fp);
	fwrite(&width_image_C3, sizeof(int), 1, fp);
	fwrite(&height_image_C3, sizeof(int), 1, fp);
	fwrite(&width_image_S4, sizeof(int), 1, fp);
	fwrite(&height_image_S4, sizeof(int), 1, fp);
	fwrite(&width_image_C5, sizeof(int), 1, fp);
	fwrite(&height_image_C5, sizeof(int), 1, fp);
	fwrite(&width_image_output, sizeof(int), 1, fp);
	fwrite(&height_image_output, sizeof(int), 1, fp);

	fwrite(&width_kernel_conv, sizeof(int), 1, fp);
	fwrite(&height_kernel_conv, sizeof(int), 1, fp);
	fwrite(&width_kernel_pooling, sizeof(int), 1, fp);
	fwrite(&height_kernel_pooling, sizeof(int), 1, fp);

	fwrite(&num_map_input, sizeof(int), 1, fp);
	fwrite(&num_map_C1, sizeof(int), 1, fp);
	fwrite(&num_map_S2, sizeof(int), 1, fp);
	fwrite(&num_map_C3, sizeof(int), 1, fp);
	fwrite(&num_map_S4, sizeof(int), 1, fp);
	fwrite(&num_map_C5, sizeof(int), 1, fp);
	fwrite(&num_map_output, sizeof(int), 1, fp);

	fwrite(&len_weight_C1, sizeof(int), 1, fp);
	fwrite(&len_bias_C1, sizeof(int), 1, fp);
	fwrite(&len_weight_S2, sizeof(int), 1, fp);
	fwrite(&len_bias_S2, sizeof(int), 1, fp);
	fwrite(&len_weight_C3, sizeof(int), 1, fp);
	fwrite(&len_bias_C3, sizeof(int), 1, fp);
	fwrite(&len_weight_S4, sizeof(int), 1, fp);
	fwrite(&len_bias_S4, sizeof(int), 1, fp);
	fwrite(&len_weight_C5, sizeof(int), 1, fp);
	fwrite(&len_bias_C5, sizeof(int), 1, fp);
	fwrite(&len_weight_output, sizeof(int), 1, fp);
	fwrite(&len_bias_output, sizeof(int), 1, fp);

	fwrite(&num_neuron_input, sizeof(int), 1, fp);
	fwrite(&num_neuron_C1, sizeof(int), 1, fp);
	fwrite(&num_neuron_S2, sizeof(int), 1, fp);
	fwrite(&num_neuron_C3, sizeof(int), 1, fp);
	fwrite(&num_neuron_S4, sizeof(int), 1, fp);
	fwrite(&num_neuron_C5, sizeof(int), 1, fp);
	fwrite(&num_neuron_output, sizeof(int), 1, fp);

	fwrite(weight_C1, sizeof(weight_C1), 1, fp);
	fwrite(bias_C1, sizeof(bias_C1), 1, fp);
	fwrite(weight_S2, sizeof(weight_S2), 1, fp);
	fwrite(bias_S2, sizeof(bias_S2), 1, fp);
	fwrite(weight_C3, sizeof(weight_C3), 1, fp);
	fwrite(bias_C3, sizeof(bias_C3), 1, fp);
	fwrite(weight_S4, sizeof(weight_S4), 1, fp);
	fwrite(bias_S4, sizeof(bias_S4), 1, fp);
	fwrite(weight_C5, sizeof(weight_C5), 1, fp);
	fwrite(bias_C5, sizeof(bias_C5), 1, fp);
	fwrite(weight_output, sizeof(weight_output), 1, fp);
	fwrite(bias_output, sizeof(bias_output), 1, fp);

	fflush(fp);
	fclose(fp);

	return true;
}

bool CNN::readModelFile(const char* name)
{
	FILE* fp = fopen(name, "rb");
	if (fp == NULL) {
		return false;
	}

	int width_image_input =0;
	int height_image_input = 0;
	int width_image_C1 = 0;
	int height_image_C1 = 0;
	int width_image_S2 = 0;
	int height_image_S2 = 0;
	int width_image_C3 = 0;
	int height_image_C3 = 0;
	int width_image_S4 = 0;
	int height_image_S4 = 0;
	int width_image_C5 = 0;
	int height_image_C5 = 0;
	int width_image_output = 0;
	int height_image_output = 0;

	int width_kernel_conv = 0;
	int height_kernel_conv = 0;
	int width_kernel_pooling = 0;
	int height_kernel_pooling = 0;

	int num_map_input = 0;
	int num_map_C1 = 0;
	int num_map_S2 = 0;
	int num_map_C3 = 0;
	int num_map_S4 = 0;
	int num_map_C5 = 0;
	int num_map_output = 0;

	int len_weight_C1 = 0;
	int len_bias_C1 = 0;
	int len_weight_S2 = 0;
	int len_bias_S2 = 0;
	int len_weight_C3 = 0;
	int len_bias_C3 = 0;
	int len_weight_S4 = 0;
	int len_bias_S4 = 0;
	int len_weight_C5 = 0;
	int len_bias_C5 = 0;
	int len_weight_output = 0;
	int len_bias_output = 0;

	int num_neuron_input = 0;
	int num_neuron_C1 = 0;
	int num_neuron_S2 = 0;
	int num_neuron_C3 = 0;
	int num_neuron_S4 = 0;
	int num_neuron_C5 = 0;
	int num_neuron_output = 0;

	fread(&width_image_input, sizeof(int), 1, fp);
	fread(&height_image_input, sizeof(int), 1, fp);
	fread(&width_image_C1, sizeof(int), 1, fp);
	fread(&height_image_C1, sizeof(int), 1, fp);
	fread(&width_image_S2, sizeof(int), 1, fp);
	fread(&height_image_S2, sizeof(int), 1, fp);
	fread(&width_image_C3, sizeof(int), 1, fp);
	fread(&height_image_C3, sizeof(int), 1, fp);
	fread(&width_image_S4, sizeof(int), 1, fp);
	fread(&height_image_S4, sizeof(int), 1, fp);
	fread(&width_image_C5, sizeof(int), 1, fp);
	fread(&height_image_C5, sizeof(int), 1, fp);
	fread(&width_image_output, sizeof(int), 1, fp);
	fread(&height_image_output, sizeof(int), 1, fp);

	fread(&width_kernel_conv, sizeof(int), 1, fp);
	fread(&height_kernel_conv, sizeof(int), 1, fp);
	fread(&width_kernel_pooling, sizeof(int), 1, fp);
	fread(&height_kernel_pooling, sizeof(int), 1, fp);

	fread(&num_map_input, sizeof(int), 1, fp);
	fread(&num_map_C1, sizeof(int), 1, fp);
	fread(&num_map_S2, sizeof(int), 1, fp);
	fread(&num_map_C3, sizeof(int), 1, fp);
	fread(&num_map_S4, sizeof(int), 1, fp);
	fread(&num_map_C5, sizeof(int), 1, fp);
	fread(&num_map_output, sizeof(int), 1, fp);

	fread(&len_weight_C1, sizeof(int), 1, fp);
	fread(&len_bias_C1, sizeof(int), 1, fp);
	fread(&len_weight_S2, sizeof(int), 1, fp);
	fread(&len_bias_S2, sizeof(int), 1, fp);
	fread(&len_weight_C3, sizeof(int), 1, fp);
	fread(&len_bias_C3, sizeof(int), 1, fp);
	fread(&len_weight_S4, sizeof(int), 1, fp);
	fread(&len_bias_S4, sizeof(int), 1, fp);
	fread(&len_weight_C5, sizeof(int), 1, fp);
	fread(&len_bias_C5, sizeof(int), 1, fp);
	fread(&len_weight_output, sizeof(int), 1, fp);
	fread(&len_bias_output, sizeof(int), 1, fp);

	fread(&num_neuron_input, sizeof(int), 1, fp);
	fread(&num_neuron_C1, sizeof(int), 1, fp);
	fread(&num_neuron_S2, sizeof(int), 1, fp);
	fread(&num_neuron_C3, sizeof(int), 1, fp);
	fread(&num_neuron_S4, sizeof(int), 1, fp);
	fread(&num_neuron_C5, sizeof(int), 1, fp);
	fread(&num_neuron_output, sizeof(int), 1, fp);

	fread(weight_C1, sizeof(weight_C1), 1, fp);
	fread(bias_C1, sizeof(bias_C1), 1, fp);
	fread(weight_S2, sizeof(weight_S2), 1, fp);
	fread(bias_S2, sizeof(bias_S2), 1, fp);
	fread(weight_C3, sizeof(weight_C3), 1, fp);
	fread(bias_C3, sizeof(bias_C3), 1, fp);
	fread(weight_S4, sizeof(weight_S4), 1, fp);
	fread(bias_S4, sizeof(bias_S4), 1, fp);
	fread(weight_C5, sizeof(weight_C5), 1, fp);
	fread(bias_C5, sizeof(bias_C5), 1, fp);
	fread(weight_output, sizeof(weight_output), 1, fp);
	fread(bias_output, sizeof(bias_output), 1, fp);

	fflush(fp);
	fclose(fp);
/*
	out2wi_S2.clear();
	out2bias_S2.clear();
	out2wi_S4.clear();
	out2bias_S4.clear();

	calc_out2wi(width_image_C1_CNN, height_image_C1_CNN, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN, out2wi_S2);
	calc_out2bias(width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN, out2bias_S2);
	calc_out2wi(width_image_C3_CNN, height_image_C3_CNN, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN, out2wi_S4);
	calc_out2bias(width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN, out2bias_S4);
*/
	return true;
}




