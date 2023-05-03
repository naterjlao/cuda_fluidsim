CUDA_ARCH=sm_86
OPENCV_ARGS=`pkg-config --cflags opencv` `pkg-config --libs opencv`
INCLUDE=./include
OBJDIR=./build
BINDIR=./bin
EXECUTABLES=$(BINDIR)/main $(BINDIR)/fluid_sim_test

################### EXECUTABLES ###################

all: $(EXECUTABLES)

$(BINDIR)/main: $(OBJDIR)/main.o $(OBJDIR)/window_utils.o $(OBJDIR)/fluid_sim.o $(OBJDIR)/fluid_utils.o
	nvcc -arch=$(CUDA_ARCH) $(OPENCV_ARGS) \
		$(OBJDIR)/main.o \
		$(OBJDIR)/window_utils.o \
		$(OBJDIR)/fluid_sim.o \
		$(OBJDIR)/fluid_utils.o \
		-o $(BINDIR)/main

$(BINDIR)/fluid_sim_test: $(OBJDIR)/fluid_sim_test.o $(OBJDIR)/fluid_sim.o $(OBJDIR)/window_utils.o $(OBJDIR)/fluid_utils.o
	nvcc -arch=$(CUDA_ARCH) \
		$(OBJDIR)/fluid_sim_test.o \
		$(OBJDIR)/fluid_sim.o \
		$(OBJDIR)/window_utils.o \
		$(OBJDIR)/fluid_utils.o \
		-o $(BINDIR)/fluid_sim_test

################### OBJECTS ###################

$(OBJDIR)/main.o: main.cu
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc main.cu -o $@

$(OBJDIR)/window_utils.o: window_utils.cu $(INCLUDE)/window_utils.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc window_utils.cu -o $@

$(OBJDIR)/fluid_sim_test.o: fluid_sim_test.cu
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_sim_test.cu -o $@

$(OBJDIR)/fluid_sim.o: fluid_sim.cu $(INCLUDE)/fluid_sim.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_sim.cu -o $@

$(OBJDIR)/fluid_utils.o: fluid_utils.cu $(INCLUDE)/fluid_utils.cuh
	nvcc -I$(INCLUDE) -arch=$(CUDA_ARCH) -rdc=true -dc fluid_utils.cu -o $@

.PHONY: all clean
clean:
	rm -rvf $(EXECUTABLES)
	rm -rvf *.o
	rm -rvf $(OBJDIR)/*.o

c: clean