/*
 * mnist.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;


int CNN::reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void CNN::readMnistImages(std::string filename, float* data_dst, int num_image)
{
	const int width_src_image = 28;
	const int height_src_image = 28;
	const int x_padding = 2;
	const int y_padding = 2;
	const float scale_min = -1;
	const float scale_max = 1;

	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	assert(number_of_images == num_image);
	file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);
	assert(n_rows == height_src_image && n_cols == width_src_image);

	int size_single_image = height_image_input_CNN * width_image_input_CNN;

	for (int i = 0; i < number_of_images; ++i) {
		int addr = size_single_image * i;
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; ++c) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				data_dst[addr + width_image_input_CNN * (r + y_padding) + c + x_padding] = (temp / 255.0) * (scale_max - scale_min) + scale_min;
			}
		}
	}
}

void CNN::readMnistLabels(std::string filename, float* data_dst, int num_image)
{
	const float scale_max = 0.8;

	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	assert(number_of_images == num_image);

	for (int i = 0; i < number_of_images; ++i) {
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));
		data_dst[i * num_map_output_CNN + temp] = scale_max;
	}
}

bool CNN::getSrcData()
{
	assert(data_input_train && data_output_train && data_input_test && data_output_test);
	std::cout << "readMNIST" << std::endl;
	std::string filename_train_images = "data/train-images-idx3-ubyte";
	std::string filename_train_labels = "data/train-labels-idx1-ubyte";
	readMnistImages(filename_train_images, data_input_train, num_patterns_train_CNN);
	readMnistLabels(filename_train_labels, data_output_train, num_patterns_train_CNN);

	std::string filename_test_images = "data/t10k-images-idx3-ubyte";
	std::string filename_test_labels = "data/t10k-labels-idx1-ubyte";
	readMnistImages(filename_test_images, data_input_test, num_patterns_test_CNN);
	readMnistLabels(filename_test_labels, data_output_test, num_patterns_test_CNN);
	return true;
}




