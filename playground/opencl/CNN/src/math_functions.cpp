/*
 * math_functions.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;


float CNN::activation_function_tanh(float x)
{
	float ep = std::exp(x);
	float em = std::exp(-x);

	return (ep - em) / (ep + em);
}

float CNN::activation_function_tanh_derivative(float x)
{
	return (1.0 - x * x);
}

float CNN::activation_function_identity(float x)
{
	return x;
}

float CNN::activation_function_identity_derivative(float x)
{
	return 1;
}

float CNN::loss_function_mse(float y, float t)
{
	return (y - t) * (y - t) / 2;
}

float CNN::loss_function_mse_derivative(float y, float t)
{
	return (y - t);
}

void CNN::loss_function_gradient(const float* y, const float* t, float* dst, int len)
{
	for (int i = 0; i < len; i++) {
		dst[i] = loss_function_mse_derivative(y[i], t[i]);
	}
}

float CNN::dot_product(const float* s1, const float* s2, int len)
{
	float result = 0.0;

	for (int i = 0; i < len; i++) {
		result += s1[i] * s2[i];
	}

	return result;
}

bool CNN::muladd(const float* src, float c, int len, float* dst)
{
	for (int i = 0; i < len; i++) {
		dst[i] += (src[i] * c);
	}
	return true;
}

void CNN::init_variable(float* val, float c, int len)
{
	for (int i = 0; i < len; i++) {
		val[i] = c;
	}
}






