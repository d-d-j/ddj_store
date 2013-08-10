################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/DDJ_StoreBuffer.cpp \
../src/DDJ_StoreController.cpp \
../src/main.cpp 

OBJS += \
./src/DDJ_StoreBuffer.o \
./src/DDJ_StoreController.o \
./src/main.o 

CPP_DEPS += \
./src/DDJ_StoreBuffer.d \
./src/DDJ_StoreController.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/parallels/INZ/PROJ/DDJ_Store/libs/pantheios/include -I/home/parallels/INZ/PROJ/DDJ_Store/libs/stlsoft/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


