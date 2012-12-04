DEBUG_CUDA := -g -G
DEBUG := -g
#DEBUG_CUDA := 
#DEBUG := 

all: emc

emc.o: emc.c
	gcc $(DEBUG) -std=c99 -c emc.c `gsl-config --cflags` -I/usr/local/cuda/include -L/usr/local/cuda/lib64 

emc_cuda.o: emc_cuda.cu
#	nvcc --compiler-bindir=/usr/bin/gcc-4.5 -m64 -c emc_cuda.cu -I/usr/local/cuda/include
#	nvcc --compiler-bindir=/usr/bin/gcc-4.4 -m64 -c emc_cuda.cu -I/usr/local/cuda/include -I${HOME}/git/libspimage/include/thrust-1.3.0 -g 
	nvcc $(DEBUG_CUDA) -c emc_cuda.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I${HOME}/git/libspimage/include/thrust-1.3.0 -arch sm_20

emc_atomic.o: emc_atomic.cu
#	nvcc --compiler-bindir=/usr/bin/gcc-4.4 -m64 -I/usr/local/cuda/include -c emc_atomic.cu -arch sm_12
	nvcc $(DEBUG_CUDA) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -c emc_atomic.cu -arch sm_20

rotations.o: rotations.c rotations.h
	gcc $(DEBUG) -std=c99 -c rotations.c

emc: emc.o emc_cuda.o emc_atomic.o rotations.o
	nvcc $(DEBUG_CUDA) -o emc emc.o emc_cuda.o emc_atomic.o rotations.o -arch sm_20 -m64 -lspimage -lconfig `gsl-config --libs`

clean:
	rm *.o emc
