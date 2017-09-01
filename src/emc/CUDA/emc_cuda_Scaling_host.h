/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_SCALING_HOST_H
#define EMC_CUDA_SCALING_HOST_H
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_update_scaling(real * d_images, int * d_mask, real * d_scaling,
                         real *d_weighted_power, int N_images, int N_slices,
                         int N_2d, real * scaling);

void cuda_update_weighted_power(real * d_images, real * d_slices, int * d_mask,
                                real * d_respons, real * d_weighted_power,
                                int N_images, int slice_start, int slice_chunk,
                                int N_2d);
void cuda_update_scaling_best(real *d_images, int *d_mask,  real *d_model,
                              real *d_scaling, real *d_weight_map,
                              real *d_respons, real *d_rotations,
                              real *x_coordinates, real *y_coordinates,
                              real *z_coordinates, int N_images, int N_slices,
                              int side, real *scaling);

void cuda_update_scaling_full(real *d_images, real *d_slices, int *d_mask,
                              real *d_scaling, real *d_weight_map, int N_2d,
                              int N_images, int slice_start, int slice_chunk,
                              enum diff_type diff);

/*
#ifdef __cplusplus
}
#endif
*/
#endif
