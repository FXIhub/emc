#include <emc_cuda_2Dpatterns.h>

void
cuda_apply_masks(real *const d_images, const int *const d_masks, const int N_2d, const int N_images) {
    int nthreads = 256;
    int nblocks = (N_2d*N_images - 1) / nthreads;
    apply_mask<<<nblocks, nthreads>>>(d_images, d_masks, N_2d*N_images);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_apply_masks): %s\n",cudaGetErrorString(status));
    }
}

void
cuda_apply_single_mask(real *const d_images, const int *const d_mask, const int N_2d, const int N_images) {
    int nthreads = 256;
    int nblocks = (N_2d*N_images - 1) / nthreads;
    apply_single_mask<<<nblocks, nthreads>>>(d_images, d_mask, N_2d, N_2d*N_images);
}

