/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_3DMODEL_H
#define EMC_CUDA_3DMODEL_H

#include "emc_cuda.h"
#include "emc_cuda_2Dpatterns.h"
#include "emc_cuda_2Dpatterns_host.h"
#include "emc_cuda_memory_host.h"
#include "emc_cuda_common.h"
#include "emc_cuda_common_host.h"
/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
__global__ void model_average_kernel(real *model, int model_size, real *average);
__global__ void cuda_divide_model_kernel(real * model, real * weight, int n);
__global__ void cuda_mask_out_model_kernel(real *model, real *weight, int n);
__global__ void get_mask_from_model(real *model, int *mask, int size);
__global__ void multiply_by_gaussian_kernel(cufftComplex *model, const real sigma);
/*
#ifdef __cplusplus 
}
#endif
*/
#endif
