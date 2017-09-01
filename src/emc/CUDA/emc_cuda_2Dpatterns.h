/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_2DPATTERNS_H
#define EMC_CUDA_2DPATTERNS_H
#include <emc_common.h>
//#ifdef __cplusplus 
 //extern "C" {
//#endif

__global__ void apply_mask(real *const array, const int *const mask, const int size);
__global__ void apply_single_mask(real * const array, const int *const mask, const int mask_size, const int size) ;
//#ifdef __cplusplus 
//}
//#endif

#endif
