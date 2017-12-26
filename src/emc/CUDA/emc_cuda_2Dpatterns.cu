#include "emc_cuda_2Dpatterns.h"

//#ifdef __cplusplus 
//extern "C" {
//#endif
__global__ 
void apply_mask(real *const array, const int *const mask, const int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        if (mask[i] == 0) {
            array[i] = -1.;
        }
    }
}

/* This function is the same as apply_mask except that it
   periodically steps throug the mask thus applying the same
   mask to multiple files. */
__global__ 
void apply_single_mask(real * const array, const int *const mask, const int mask_size, const int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        if (mask[i%mask_size] == 0) {
            array[i] = -1;
        }
    }
}

__global__
void apply_single_mask_zeros(real * const array, const int *const mask, const int mask_size, const int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        if (mask[i%mask_size] == 0) {
            array[i] =0;
        }
    }
}


//#ifdef __cplusplus 
//}
//#endif
