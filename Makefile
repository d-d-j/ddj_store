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

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM21    := -gencode arch=compute_20,code=sm_21
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM21) $(GENCODE_SM30)

SRC_FILES := $(wildcard src/*.cpp src/*/*.cpp src/*/*.cu)

OBJS := $(SRC_FILES:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

all: DDJ_Store

debug: COMPILER += -DDEBUG -g
debug: all

src/%.o: ./src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: $(COMPILER)'
	$(COMPILER) $(DEFINES) $(INCLUDES) $(WARNINGS_ERRORS) -c -g $(STANDART) -MMD -MP -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ./src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc $(GENCODE_FLAGS) $(INCLUDES) -c -g -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

run: all
	./DDJ_Store

DDJ_Store: $(OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(COMPILER) -o "DDJ_Store" $(OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

test: all
	./DDJ_Store --test 2> /dev/null
	
leak-check: all
	valgrind $(VALGRIND_OPTIONS) ./DDJ_Store --test

check:
	cppcheck --enable=all -j 4 -q ./src/

clean:
	-$(RM) $(OBJS) DDJ_Store
	-@echo ' '

.PHONY: all clean dependents
