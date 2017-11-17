/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_SCALING_H
#define EMC_CUDA_SCALING_H
#include <emc_cuda.h>
#include <emc_cuda_ec.h>
#include <emc_cuda_Respons.h>
/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
// device functions

__device__ real calculate_scaling_poisson(real *image, real *slice, int *mask,
                               real *weight_map, int N_2d, int tid, int step);
__device__ real calculate_scaling_absolute(real *image, real *slice, int *mask,
                                real *weight_map, int N_2d, int tid, int step);

__device__ real calculate_scaling_relative(real *image, real *slice, int *mask,
                                real *weight_map, int N_2d, int tid, int step);
__device__ real calculate_scaling_true_poisson(real *image, real *slice, int *mask,
                                real *weight_map, int N_2d, int tid, int step);


// global functions
__global__ void slice_weighting_kernel(real * images,int * mask, real * scaling,
                            real *weighted_power, int N_slices, int N_2d);

__global__ void calculate_weighted_power_kernel(real * images, real * slices, int * mask,
                                     real *respons, real * weighted_power,
                                     int N_images, int slice_start,
                                     int slice_chunk, int N_2d);

__global__ void update_scaling_best_kernel(real *scaling, real *images, real *model,
                                int *mask, real *weight_map, real *rotations,
                                real *x_coordinates, real *y_coordinates,
                                real *z_coordinates, int side, int *best_rotation);

__global__ void update_scaling_full_kernel(real *images, real *slices, int *mask,
                                real *scaling, real *weight_map, int N_2d,
 
                                int slice_start, enum diff_type diff);

/*
#ifdef __cplusplus 
}
#endif
*/
#endif



