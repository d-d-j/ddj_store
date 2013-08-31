RM := rm -rf

CPP_SRCS += \
./src/BTreeMonitor.cpp \
./src/StoreBuffer.cpp \
./src/StoreController.cpp \
./src/GpuUploaderMonitor.cpp \
./src/main.cpp

OBJS += \
./src/BTreeMonitor.o \
./src/StoreBuffer.o \
./src/StoreController.o \
./src/GpuUploaderMonitor.o \
./src/main.o 

CPP_DEPS += \
./src/BTreeMonitor.d \
./src/StoreBuffer.d \
./src/StoreController.d \
./src/GpuUploaderMonitor.d \
./src/main.d


LIBS := -L"./libs/pantheios/lib" -lpantheios.1.core.gcc46 -lboost_thread-mt -lpantheios.1.be.fprintf.gcc46 -lpantheios.1.bec.fprintf.gcc46 -lpantheios.1.fe.all.gcc46 -lpantheios.1.util.gcc46
INCLUDES := -I"./libs/pantheios/include" -I"./libs/stlsoft/include"
WARNINGS_ERRORS := -pedantic -pedantic-errors -Wall -Wextra -Wno-deprecated -Werror
STANDART := -std=c++11

src/%.o: ./src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++  $(INCLUDES) $(WARNINGS_ERRORS) -c $(STANDART) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
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
