CUDA_ARCH=sm_86
OPENCV_ARGS=`pkg-config --cflags opencv` `pkg-config --libs opencv`
INCLUDE=./include
EXECUTABLES=main fluid_sim_test

################### EXECUTABLES ###################

all: $(EXECUTABLES)

main: main.o window_utils.o fluid_sim.o fluid_utils.o
	nvcc -arch=$(CUDA_ARCH) $(OPENCV_ARGS) \
		main.o \
		window_utils.o \
		fluid_sim.o \
		fluid_utils.o \
		-o main

fluid_sim_test: fluid_sim_test.o fluid_sim.o window_utils.o fluid_utils.o
	nvcc -arch=$(CUDA_ARCH) \
		fluid_sim_test.o \
		fluid_sim.o \
		window_utils.o \
		fluid_utils.o \
		-o fluid_sim_test

################### OBJECTS ###################

main.o: main.cu
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc main.cu

window_utils.o: window_utils.cu $(INCLUDE)/window_utils.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc window_utils.cu

fluid_sim_test.o: fluid_sim_test.cu
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_sim_test.cu

fluid_sim.o: fluid_sim.cu $(INCLUDE)/fluid_sim.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_sim.cu

fluid_utils.o: fluid_utils.cu $(INCLUDE)/fluid_utils.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_utils.cu

.PHONY: all clean
clean:
	rm -rvf $(EXECUTABLES)
	rm -rvf *.o

c: clean