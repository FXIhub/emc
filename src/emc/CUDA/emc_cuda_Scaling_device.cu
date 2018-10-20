/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_Scaling.h>
#include <emc_cuda_common.h>


__device__
real calculate_scaling_poisson(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step)
{
    __shared__ real sum_cache[256];
    __shared__ real weight_cache[256];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10) {
            sum_cache[tid] += image[i]*image[i]/slice[i]*weight_map[i];
            weight_cache[tid] += image[i]*weight_map[i];
        }
    }
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    __syncthreads();
    return sum_cache[0] / weight_cache[0];
}


__device__
real calculate_scaling_absolute(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step){
    __shared__ real sum_cache[256];
    __shared__ real weight_cache[256];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10 && image[i] > 1.e-10) {
            sum_cache[tid] += image[i]*image[i]*weight_map[i];
            weight_cache[tid] += image[i]*slice[i]*weight_map[i];
        }
    }
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    __syncthreads();
    return sum_cache[0] / weight_cache[0];
}

__device__ real calculate_scaling_relative(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step){
    __shared__ real sum_cache[256];
    __shared__ real weight_cache[256];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10) {
            sum_cache[tid] += image[i]*image[i]/(slice[i]*slice[i]) * weight_map[i];
            weight_cache[tid] += image[i]/slice[i] * weight_map[i];
        }
    }
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    __syncthreads();
    return sum_cache[0] / weight_cache[0];
}


__device__ real calculate_scaling_true_poisson(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step){
    __shared__ real sum_cache[256];
    __shared__ real weight_cache[256];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0.;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] > 0 && slice[i] > 1.e-10 && image[i] >1e-10) {        
        //if (mask[i] > 0) {
            //sum_cache[tid] += (image[i])* (image[i]);
            //weight_cache[tid] += slice[i]* (image[i]);
            sum_cache[tid] += image[i];
            weight_cache[tid] += slice[i];
        }
    }
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    __syncthreads();
    
    return weight_cache[0] >0? sum_cache[0]/weight_cache[0]*1.0f:0;
}
