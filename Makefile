all: emc

emc.o: emc.c emc.h
	gcc -g -std=c99 -c emc.c `gsl-config --cflags` -I/usr/local/cuda/include/

emc_cuda.o: emc_cuda.cu
#	nvcc -m64 -c emc_cuda.cu 
	nvcc -m64 -c emc_cuda.cu -I${HOME}/git/libspimage/include/thrust-1.3.0 -g

emc_atomic.o: emc_atomic.cu
	nvcc -m64 -c emc_atomic.cu -arch sm_12

rotations.o: rotations.c rotations.h
	gcc -g -std=c99 -c rotations.c

emc: emc.o emc_cuda.o emc_atomic.o rotations.o
	nvcc -o emc emc.o emc_cuda.o emc_atomic.o rotations.o -m64 -lspimage -lconfig `gsl-config --libs` -L/opt/cuda/lib

clean:
	rm *.o emc
