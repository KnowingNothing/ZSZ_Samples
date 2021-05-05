#include "cnn.h"

using namespace std;

void CNN::init()
{
	//初始化数据
	int len1 = width_image_input_CNN * height_image_input_CNN * num_patterns_train_CNN;
	data_input_train = new float[len1];
	init_variable(data_input_train, -1.0, len1);

	int len2 = num_map_output_CNN * num_patterns_train_CNN;
	data_output_train = new float[len2];
	init_variable(data_output_train, -0.8, len2);

	int len3 = width_image_input_CNN * height_image_input_CNN * num_patterns_test_CNN;
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
}

float CNN::uniform_rand(float min, float max)
{
	//std::mt19937 gen(1);
	std::random_device rd;
    std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dst(min, max);
	return dst(gen);
}

bool CNN::uniform_rand(float* src, int len, float min, float max)
{
	for (int i = 0; i < len; i++) {
		src[i] = uniform_rand(min, max);
	}
	return true;
}

bool CNN::initWeightThreshold()
{
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





