/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/

#ifndef EMC_CUDA_EC_H
#define EMC_CUDA_EC_H

#include <emc_cuda.h>
#include <emc_cuda_common.h>
#include <emc_common.h>
/*#ifdef __cplusplus 
extern "C" {
#endif
*/
//device functions
__device__ void cuda_get_slice(real *model, real *slice,
                    real *rot, real *x_coordinates,
                    real *y_coordinates, real *z_coordinates, int slice_rows,
                    int slice_cols, int model_x, int model_y, int model_z,
                    int tid, int step);
__device__ real interpolate_model_get(real *model, int model_x, int model_y,
                           int model_z, real new_x, real new_y, real new_z) ;


__device__ void cuda_get_slice_interpolate(real *model, real *slice, real *rot,
                                real *x_coordinates, real *y_coordinates, real *z_coordinates,
                                int slice_rows, int slice_cols, int model_x, int model_y, int model_z,
                                int tid, int step);

__device__ void cuda_insert_slice_interpolate(real *model, real *weight, real *slice,
                                   int * mask, real w, real *rot, real *x_coordinates,
                                   real *y_coordinates, real *z_coordinates, int slice_rows,
                                   int slice_cols, int model_x, int model_y, int model_z,
                                   int tid, int step);

__device__ void interpolate_model_set(real *model, real *model_weight,
                           int model_x, int model_y, int model_z,
                           real new_x, real new_y, real new_z,
                           real value, real value_weight);
__device__ void cuda_insert_slice(real *model, real *weight, real *slice,
                       int * mask, real w, real *rot, real *x_coordinates,
                       real *y_coordinates, real *z_coordinates, int slice_rows,
                       int slice_cols, int model_x, int model_y, int model_z,
                       int tid, int step);
//global functions

__global__ void get_slices_kernel(real * model, real * slices, real *rot, real *x_coordinates,
                       real *y_coordinates, real *z_coordinates, int slice_rows,
                       int slice_cols, int model_x, int model_y, int model_z,
                       int start_slice);

__global__ void cuda_test_interpolate_kernel(real *model, int side, real *return_value);

__global__ void cuda_test_interpolate_set_kernel(real *model, real *weight, int side) ;

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
                          real * scaling, int N_images, int N_2d,
                          real * slices_total_respons, real * rot,
                          real * x_coord, real * y_coord, real * z_coord,
                          real * model, real * weight,
                          int slice_rows, int slice_cols,
                          int model_x, int model_y, int model_z);

//host functions
/*
#ifdef __cplusplus 
}
#endif
*/
#endif
