main: main.cpp
	nvcc `pkg-config --cflags opencv` `pkg-config --libs opencv` main.cpp -o main

:PHONY=clean
clean:
	rm main
