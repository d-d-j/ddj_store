RM := rm -rf
OS := $(shell uname)

ifeq ($(OS),Darwin)
	COMPILER := clang++
	LIBS := -L"/usr/local/cuda/lib" -lcudart -lboost_system -lboost_thread -lpthread -lboost_thread -lboost_program_options -llog4cplus -lgtest -lgtest_main
	STANDART := -std=c++11 -stdlib=libc++
else
	#you can use g++ or clang or color-gcc
	COMPILER := g++
	LIBS := -L"/usr/local/cuda/lib64" -lcudart -lboost_system -lboost_thread -lpthread -lboost_thread -lboost_program_options -llog4cplus -lgtest -lgtest_main
	STANDART := -std=c++0x
endif

INCLUDES := -I"/usr/local/cuda/include"
DEFINES := -D __GXX_EXPERIMENTAL_CXX0X__
WARNINGS_ERRORS := -pedantic -Wall -Wextra -Wno-deprecated -Wno-unused-parameter  -Wno-enum-compare

VALGRIND_OPTIONS = --tool=memcheck --leak-check=yes -q

OBJS += \
./src/UnitTests/StoreQueryCoreTest.o \
./src/UnitTests/SemaphoreTest.o \
./src/UnitTests/BTreeMonitorTest.o \
./src/BTree/BTreeMonitor.o \
./src/Core/Config.o \
./src/Core/Semaphore.o \
./src/Cuda/CudaCommons.o \
./src/Cuda/CudaController.o \
./src/Network/NetworkClient.o \
./src/Store/StoreBuffer.o \
./src/Store/StoreController.o \
./src/Store/StoreQueryCore.o \
./src/Store/StoreUploadCore.o \
./src/Store/StoreQuery.o \
./src/Task/Task.o \
./src/Task/TaskMonitor.o \
./src/Node.o \
./src/main.o

CPP_DEPS += \
./src/UnitTests/StoreQueryCoreTest.d \
./src/UnitTests/SemaphoreTest.d \
./src/UnitTests/BTreeMonitorTest.d \
./src/BTree/BTreeMonitor.d \
./src/Core/Config.d \
./src/Core/Semaphore.d \
./src/Cuda/CudaCommons.d \
./src/Cuda/CudaController.d \
./src/Network/NetworkClient.d \
./src/Store/StoreBuffer.d \
./src/Store/StoreController.d \
./src/Store/StoreQueryCore.d \
./src/Store/StoreUploadCore.d \
./src/Store/StoreQuery.d \
./src/Task/Task.d \
./src/Task/TaskMonitor.d \
./src/Node.d \
./src/main.d

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
	GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
	GENCODE_SM20    := -gencode arch=compute_20,code=sm_21
	GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
	GENCODE_FLAGS   := $(GENCODE_SM21) $(GENCODE_SM30)

src/%.o: ./src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: $(COMPILER)'
	$(COMPILER) $(DEFINES) $(INCLUDES) $(WARNINGS_ERRORS) -c -g $(STANDART) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ./src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc $(GENCODE_FLAGS) $(INCLUDES) -c -g -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: DDJ_Store

DDJ_Store: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(COMPILER) -o "DDJ_Store" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

test: all
	./DDJ_Store --test 2> /dev/null
leak-check: all
	valgrind $(VALGRIND_OPTIONS) ./DDJ_Store --test

check:
	cppcheck --enable=all -j 4 -q ./src/

clean:
	-$(RM) $(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) DDJ_Store
	-@echo ' '

.PHONY: all clean dependents
