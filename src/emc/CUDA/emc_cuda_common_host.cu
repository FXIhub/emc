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

void cuda_matrix_scalar(real* d_matrix, int N_images, int N_slices, real d_scalar){
    int nblocks = N_images;
    int nthreads= TNUM;
    // printf("In cuda_max_vector  %d %d\n", N_images,N_slices);
    cuda_matrix_scalar_kernel <<<nblocks, nthreads>>>(d_matrix,N_images,N_slices,d_scalar);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_matrix_scalar): %s\n",cudaGetErrorString(status));
    }
}

