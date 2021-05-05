#include "cnn.h"

using namespace std;

struct  timeval tsBegin, tsEnd, ToltsBegin, ToltsEnd;
long  t1Duration;

int CNN::get_index(int x, int y, int channel, int width, int height, int depth)
{
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	assert(channel >= 0 && channel < depth);
	return (height * channel + y) * width + x;
}

bool CNN::train()
{
	std::cout << "training" << std::endl;
	int iter = 0;
	for (iter = 0; iter < num_epochs_CNN; iter++) {
		std::cout << "epoch: " << iter + 1 << std::endl;
		gettimeofday(&ToltsBegin, NULL);
		for (int i = 0; i < num_patterns_train_CNN; i++) {

			if (i % 1000 == 0) {
				gettimeofday(&tsBegin, NULL);
			}
			//1 输入模式顺传播
			data_single_image = data_input_train + i * num_neuron_input_CNN;
			data_single_label = data_output_train + i * num_neuron_output_CNN;

			memcpy(neuron_input, data_single_image, num_neuron_input_CNN*sizeof(float));

			Forward_C1();
			Forward_S2();
			Forward_C3();
			Forward_S4();
			Forward_C5();
			Forward_output();

			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				printf("%dth --> fordward: %1d ms, ", i, t1Duration);
				gettimeofday(&tsBegin, NULL);
			}

			//2 输出误差逆传播
			Backward_output();
			Backward_C5();
			Backward_S4();
			Backward_C3();
			Backward_S2();
			Backward_C1();
			Backward_input();

			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				printf("backward: %1d ms, ", t1Duration);
				gettimeofday(&tsBegin, NULL);
			}

			UpdateWeights();

			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				printf(" UpdateWeights: %1d ms\n", t1Duration);
			}
		}   //3 循环记忆训练
		//4 学习结果判别
		float accuracyRate = test();
		std::cout << ",    accuray rate: " << accuracyRate << std::endl;
		if (accuracyRate > accuracy_rate_CNN) {
			saveModelFile("cnn.model");
			std::cout << "generate cnn model" << std::endl;
			break;
		}
		saveModelFile("cnn.model");
		std::cout << "generate cnn model" << std::endl;
		gettimeofday(&ToltsEnd, NULL);
		t1Duration = 1000000L * (ToltsEnd.tv_sec - ToltsBegin.tv_sec) + (ToltsEnd.tv_usec - ToltsBegin.tv_usec);
		printf(" *******  every epoch : %1d s ^_^ \n", t1Duration/1000000L);
	}

	if (iter == num_epochs_CNN) {
		saveModelFile("cnn.model");
		std::cout << "generate cnn model" << std::endl;
	}
    return true;
}

void CNN::update_weights_bias(const float* delta, float* e_weight, float* weight, int len)
{
	for (int i = 0; i < len; i++) {
		e_weight[i] += delta[i] * delta[i];
		weight[i] -= learning_rate_CNN * delta[i] / (std::sqrt(e_weight[i]) + eps_CNN);
	}
}

bool CNN::UpdateWeights()
{
	update_weights_bias(delta_weight_C1, E_weight_C1, weight_C1, len_weight_C1_CNN);
	update_weights_bias(delta_bias_C1, E_bias_C1, bias_C1, len_bias_C1_CNN);

	update_weights_bias(delta_weight_S2, E_weight_S2, weight_S2, len_weight_S2_CNN);
	update_weights_bias(delta_bias_S2, E_bias_S2, bias_S2, len_bias_S2_CNN);

	update_weights_bias(delta_weight_C3, E_weight_C3, weight_C3, len_weight_C3_CNN);
	update_weights_bias(delta_bias_C3, E_bias_C3, bias_C3, len_bias_C3_CNN);

	update_weights_bias(delta_weight_S4, E_weight_S4, weight_S4, len_weight_S4_CNN);
	update_weights_bias(delta_bias_S4, E_bias_S4, bias_S4, len_bias_S4_CNN);

	update_weights_bias(delta_weight_C5, E_weight_C5, weight_C5, len_weight_C5_CNN);
	update_weights_bias(delta_bias_C5, E_bias_C5, bias_C5, len_bias_C5_CNN);

	update_weights_bias(delta_weight_output, E_weight_output, weight_output, len_weight_output_CNN);
	update_weights_bias(delta_bias_output, E_bias_output, bias_output, len_bias_output_CNN);

	return true;
}

float CNN::test()
{
	int count_accuracy = 0;

	for (int num = 0; num < num_patterns_test_CNN; num++) {
		data_single_image = data_input_test + num * num_neuron_input_CNN;
		data_single_label = data_output_test + num * num_neuron_output_CNN;

		memcpy(neuron_input, data_single_image, num_neuron_input_CNN*sizeof(float));

		Forward_C1();
		Forward_S2();
		Forward_C3();
		Forward_S4();
		Forward_C5();
		Forward_output();

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




