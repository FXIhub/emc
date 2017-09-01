/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/
#ifndef EMC_CUDA_WEIGHT_H
#define EMC_CUDA_WEIGHT_H
#include <emc_cuda.h>
#include <emc_common.h>
/*#ifdef __cplusplus 
extern "C" {
#endif
*/
__global__ void calculate_weight_map_inner_kernel(real *weight_map, real width, real falloff) ;

__global__ void calculate_weight_map_ring_kernel(real *weight_map, real inner_rad, real inner_falloff, real outer_rad, real outer_falloff);
/*#ifdef __cplusplus 
}
#endif
*/
#endif
