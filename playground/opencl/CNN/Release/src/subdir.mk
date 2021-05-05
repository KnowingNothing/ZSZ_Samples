################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/backward.cpp \
../src/bmp.cpp \
../src/cnn.cpp \
../src/forward.cpp \
../src/init.cpp \
../src/main.cpp \
../src/math_functions.cpp \
../src/mnist.cpp \
../src/model.cpp \
../src/predict.cpp \
../src/train.cpp 

OBJS += \
./src/backward.o \
./src/bmp.o \
./src/cnn.o \
./src/forward.o \
./src/init.o \
./src/main.o \
./src/math_functions.o \
./src/mnist.o \
./src/model.o \
./src/predict.o \
./src/train.o 

CPP_DEPS += \
./src/backward.d \
./src/bmp.d \
./src/cnn.d \
./src/forward.d \
./src/init.d \
./src/main.d \
./src/math_functions.d \
./src/mnist.d \
./src/model.d \
./src/predict.d \
./src/train.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++11 -Wall -c -O3 -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


