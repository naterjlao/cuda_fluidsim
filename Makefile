CUDA_ARCH=sm_86
INCLUDE=./include

################### EXECUTABLES ###################

main: main.cu gradient.o
	nvcc gradient.o `pkg-config --cflags opencv` `pkg-config --libs opencv` main.cu -o main

fluid_sim_test: fluid_sim_test.o fluid_sim.o
	nvcc -arch=$(CUDA_ARCH) fluid_sim_test.o fluid_sim.o -o fluid_sim_test

################### OBJECTS ###################

gradient.o: gradient.cu $(INCLUDE)/gradient.hpp
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc gradient.cu

fluid_sim_test.o: fluid_sim_test.cu
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_sim_test.cu

fluid_sim.o: fluid_sim.cu $(INCLUDE)/fluid_sim.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_sim.cu

.PHONY: clean
clean:
	rm -rvf main
	rm -rvf fluid_sim_test
	rm -rvf *.o

c: clean