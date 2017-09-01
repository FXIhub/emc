/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_common.h>

__global__ void cuda_calculate_max_vectors_kernel(real* respons, int N_images, int N_slices, real* d_maxr){
    __shared__ real cache[256];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_slice = tid;i_slice < N_slices;i_slice += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    __syncthreads();
    inblock_maximum(cache);
    real max = cache[0];
    d_maxr[i_image]=max;
}

__global__
void cuda_vector_divide_kernel(real * nom, real * den, int n){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        if(den[i] > 0.0f){
            nom[i] /= den[i];
        }else{
            nom[i] = -1.f;
        }
    }
    __syncthreads();
}
