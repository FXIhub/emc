//
#include "emc_cuda_common.h"
void cuda_max_vector(real* d_matrix, int N_images, int N_slices, real* d_maxr){
    int nblocks = N_images;
    int nthreads= TNUM;
    // printf("In cuda_max_vector  %d %d\n", N_images,N_slices);
    cuda_calculate_max_vectors_kernel <<<nblocks, nthreads>>>(d_matrix,N_images,N_slices,d_maxr);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (Res max vector): %s\n",cudaGetErrorString(status));
    }
}

void cuda_vector_divide(real* nom, real* den, int N){
    int nblocks = N;
    int nthreads= TNUM;
    cuda_vector_divide_kernel <<<nblocks, nthreads>>> (nom, den, N);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (Res max vector): %s\n",cudaGetErrorString(status));
    }
}

