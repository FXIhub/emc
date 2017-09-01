/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_RESPONS_H
#define EMC_CUDA_RESPONS_H
#include <emc_cuda.h>
/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
//device functions
__device__ void cuda_calculate_responsability_absolute(float *slice, float *image, int *mask, real *weight_map,
                                            real sigma, real scaling, int N_2d, int tid, int step,
                                            real * sum_cache, real *count_cache);
__device__ void cuda_calculate_responsability_relative(float *slice, float *image, int *mask, real *weight_map,
                                            real sigma, real scaling, int N_2d, int tid, int step,
                                            real *sum_cache, real *count_cache);
__device__ void cuda_calculate_responsability_poisson(float *slice, float *image, int *mask, real *weight_map,
                                           real sigma, real scaling, int N_2d, int tid, int step,
                                           real * sum_cache, real * count_cache);
__device__ void cuda_calculate_responsability_true_poisson(float *slice, float *image,
                                                int *mask, real sigma, real scaling, real *weight_map,
                                                int N_2d, int tid, int step,
                                                real * sum_cache, real * count_cache);

__device__ void cuda_calculate_responsability_annealing_poisson(float *slice, float *image, int *mask, real sigma,
                                                     real scaling, real *weight_map, int N_2d, int tid,
                                                     int step, real *sum_cache, real *count_cache) ;

__device__ void cuda_calculate_responsability_absolute_atomic(float *slice,  float *image, int *mask, real sigma,
                                                   real scaling, int N_2d, int tid, int step,
                                                   real * sum_cache, int * count_cache);
__device__ void cuda_calculate_responsability_poisson_atomic(float *slice, float *image,
                                                  int *mask, real sigma, real scaling,
                                                  int N_2d, int tid, int step,
                                                  real * sum_cache, int * count_cache);
__device__ void cuda_calculate_responsability_true_poisson_atomic(float *slice, float *image,
                                                       int *mask, real sigma, real scaling,
                                                       int N_2d, int tid, int step,
                                                       real * sum_cache, int * count_cache);

// global functions
__global__ void calculate_responsabilities_kernel(float * slices, float * images, int * mask, real *weight_map,
                                       real sigma, real * scaling, real * respons, real *weights,
                                       int N_2d, int slice_start, enum diff_type diff);
__global__ void calculate_best_rotation_kernel(real *respons, real* best_respons, int *best_rotation, int N_slices);
__global__ void cuda_normalize_responsabilities_single_kernel(real *respons, int N_slices, int N_images);
__global__ void cuda_normalize_responsabilities_uniform_kernel(real * respons, int N_slices, int N_images);

__global__ void cuda_normalize_responsabilities_kernel(real * respons, int N_slices, int N_images);

__global__ void collapse_responsabilities_kernel(real *respons, int N_slices);
__global__ void cuda_respons_max_expf_kernel(real* respons,real* d_tmp,real* max,int N_slices,int N_images, real* d_sum);
__global__ void cuda_norm_respons_sumexpf_kernel(real * respons,  real* d_sum, real* max, int N_images, int allocate_slices);

#endif
