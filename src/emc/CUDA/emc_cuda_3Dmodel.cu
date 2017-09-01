/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/

#include "emc_cuda_3Dmodel.h"
#include "emc_cuda_common.h"

//#ifdef __cplusplus 
//extern "C" {
//#endif

__global__
void model_average_kernel(real *model, int model_size, real *average) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    //const int i1 = blockIdx.x;
    __shared__ real sum_cache[256];
    __shared__ int weight_cache[256];
    sum_cache[tid] = 0.;
    weight_cache[tid] = 0;
    for (int i = tid; i < model_size; i+=step) {
        if (model[i] >= 0.) {
            sum_cache[tid] += model[i];
            weight_cache[tid] += 1;
        }
    }
    inblock_reduce(sum_cache);
    inblock_reduce(weight_cache);
    __syncthreads();
    if (tid == 0) {
        *average = sum_cache[0] / weight_cache[0];
    }
}


__global__
void cuda_divide_model_kernel(real * model, real * weight, int n){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        if(weight[i] > 0.0f){
            model[i] /= weight[i];
        }else{
            model[i] = -1.f;
        }
    }
}

__global__
void cuda_mask_out_model_kernel(real *model, real *weight, int n){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        if(weight[i] <= 0.0f){
            model[i] = -1.0f;
        }
    }
}


__global__ void get_mask_from_model(real *model, int *mask, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size) {
        if (model[i] < 0.) {
            mask[i] = 0;
            model[i] = 0.;
        } else {
            mask[i] = 1;
        }
    }
}


__global__
void multiply_by_gaussian_kernel(cufftComplex *model, const real sigma) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int y = blockIdx.x;
    const int z = blockIdx.y;
    const int model_side = gridDim.x;

    real radius2;
    real sigma2 = pow(sigma/(real)model_side, 2);
    int dx, dy, dz;
    if (model_side - y < y) {
        dy = model_side - y;
    } else {
        dy = y;
    }
    if (model_side - z < z) {
        dz = model_side - z;
    } else {
        dz = z;
    }

    for (int x = tid; x < (model_side/2+1); x += step) {
        if (model_side - x < x) {
            dx = model_side - x;
        } else {
            dx = x;
        }
        // find distance to top left
        radius2 = pow((real)dx, 2) + pow((real)dy, 2) + pow((real)dz, 2);
        // calculate gaussian kernel
        model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
        model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));

    }
}

//#ifdef __cplusplus 
//}
//#endif
