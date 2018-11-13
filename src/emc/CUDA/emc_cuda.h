/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_H
#define EMC_CUDA_H
/*#ifdef __cplusplus 
extern "C" {
#endif
*/
#include <emc_common.h>


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <cufft.h>
#include <emc_cuda_common.h>
#include <emc_cuda_common_host.h>

#define TNUM 256

const real min_resp = -86520.;
const real min_tol = 1e-10;
#define INTERPOLATION_METHOD 1//  0 = NEAREST 1 = LINEAR  2=MIX CUBIC LINEAR

/*
#ifdef __cplusplus 
}
#endif
*/
#endif
