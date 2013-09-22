RM := rm -rf

OBJS += \
./src/BTree/BTreeMonitor.o \
./src/Store/StoreBuffer.o \
./src/Store/StoreController.o \
./src/GpuUpload/GpuUploadMonitor.o \
./src/GpuUpload/GpuUploadCore.o \
./src/Network/Server.o \
./src/Query/QueryMonitor.o \
./src/Query/QueryCore.o \
./src/Task/TaskResult.d \
./src/Task/StoreTask.d \
./src/Task/StoreTaskMonitor.d \
./src/main.o 

CPP_DEPS += \
./src/BTree/BTreeMonitor.d \
./src/Store/StoreBuffer.d \
./src/Store/StoreController.d \
./src/GpuUpload/GpuUploadMonitor.d \
./src/GpuUpload/GpuUploadCore.d \
./src/Network/Server.d \
./src/Query/QueryMonitor.d \
./src/Query/QueryCore.d \
./src/Task/TaskResult.d \
./src/Task/StoreTask.d \
./src/Task/StoreTaskMonitor.d \
./src/main.d


LIBS := -L"/usr/local/cuda/lib64" -lcudart -L"./libs/pantheios/lib" -lpantheios.1.core.gcc46 -lpantheios.1.be.fprintf.gcc46 -lpantheios.1.bec.fprintf.gcc46 -lpantheios.1.fe.all.gcc46 -lpantheios.1.util.gcc46 -lboost_system -lboost_thread -lpthread -lboost_thread-mt

INCLUDES := -I"/usr/local/cuda/include" -I"./libs/pantheios/include" -I"./libs/stlsoft/include"
WARNINGS_ERRORS := -pedantic -pedantic-errors -Wall -Wextra -Wno-deprecated -Werror
STANDART := -std=c++0x
DEFINES := -D __GXX_EXPERIMENTAL_CXX0X__

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
	GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
	GENCODE_SM20    := -gencode arch=compute_20,code=sm_21
	GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
	GENCODE_FLAGS   := $(GENCODE_SM21) $(GENCODE_SM30)

src/%.o: ./src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++  $(DEFINES) $(INCLUDES) $(WARNINGS_ERRORS) -c $(STANDART) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ./src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc $(GENCODE_FLAGS) -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
	
all: DDJ_Store

DDJ_Store: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++  -o "DDJ_Store" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	-$(RM) $(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) DDJ_Store
	-@echo ' '

.PHONY: all clean dependents
