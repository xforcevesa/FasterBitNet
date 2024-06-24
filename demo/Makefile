CXX=nvcc
LFLAGS=-lcublas
IFLAGS=-I/usr/local/cuda/include
CXXFLAGS=-std=c++11 -O3 $(IFLAGS) $(LFLAGS)

all: cublas_fp16_demo

cublas_fp16_demo: cublas_fp16_demo.cu
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	rm -f cublas_fp16_demo a.out

