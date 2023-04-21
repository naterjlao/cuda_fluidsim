DisplayImage: DisplayImage.cpp
	g++ `pkg-config --cflags opencv` `pkg-config --libs opencv` DisplayImage.cpp -o DisplayImage

:PHONY=clean
clean:
	rm DisplayImage
