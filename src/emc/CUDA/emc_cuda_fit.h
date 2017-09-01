/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */
#ifndef EMC_CUDA_FIT_H
#define EMC_CUDA_FIT_H

#include <emc_cuda.h>
//Device functions
/*
#ifdef __cplusplus 
extern "C" {
#endif
*/

__device__
void cuda_calculate_fit_error(float* slice, float *image,
int *mask, real scaling,
int N_2d, int tid, int step,
real *sum_cache, int *count_cache);
__device__
void cuda_calculate_fit2_error(float* slice, float *image,
int *mask, real scaling,
int N_2d, int tid, int step,
real *nom_cache, real *den_cache);

__device__
real cuda_calculate_single_fit_error(float* slice, float *image,
int *mask, real scaling,
int N_2d, int tid, int step) ;
__device__
real cuda_calculate_single_fit2_error(float* slice, float *image,
int *mask, real scaling,
int N_2d, int tid, int step);

//Global functions
__global__
void calculate_fit_kernel(real *slices, real *images, int *mask,
real *respons, real *fit, real sigma,
real *scaling, int N_2d, int slice_start);

__global__
void calculate_fit_best_rot_kernel(real *slices, real *images, int *mask,
int *best_rot, real *fit,
real *scaling, int N_2d, int slice_start);

__global__
void calculate_fit_best_rot_local_kernel(real *slices, real *images, int *mask,
                                   int *best_rot, real *fit,
                                   real *scaling, int N_2d, int slice_start,
                                         int offset);

__global__
void calculate_fit_best_rot2_kernel(real *slices, real *images, int *mask,
int *best_rot, real *fit,
real *scaling, int N_2d, int slice_start);
__global__
void calculate_radial_fit_kernel(real * slices , real * images, int * mask,
real * respons, real * scaling, real * radial_fit,
real * radial_fit_weight, real * radius,
int N_2d,  int side,  int slice_start);
/*__global__
void calculate_fit_local_kernel(real *slices, real *images, int *mask,
                                real *respons, real *fit, real sigma,
                                real *scaling, int N_2d, int slice_start, real* nom, real* den);
 */
/*
#ifdef __cplusplus 
}
#endif
*/
#endif
