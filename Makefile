INCLUDE=./include

main: main.cu gradient.o
	nvcc gradient.o `pkg-config --cflags opencv` `pkg-config --libs opencv` main.cu -o main

gradient.o: gradient.cu $(INCLUDE)/gradient.hpp
	nvcc -I$(INCLUDE) -c gradient.cu

.PHONY: clean
clean:
	rm -rvf main
	rm -rvf *.o

c: clean