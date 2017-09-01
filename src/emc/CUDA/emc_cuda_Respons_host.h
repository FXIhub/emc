/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */


#ifndef EMC_CUDA_RESPONS_HOST_H
#define EMC_CUDA_RESPONS_HOST_H

void cuda_calculate_responsabilities(real * d_slices, real * d_images, int * d_mask, real *d_weight_map,
                                     real sigma, real * d_scaling, real * d_respons, real *d_weights,
                                     int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff);
void cuda_calculate_responsabilities_sum(real * respons, real * d_respons, int N_slices,
                                         int N_images);
void cuda_normalize_responsabilities_single(real *d_respons, int N_slices, int N_images);
void cuda_normalize_responsabilities(real * d_respons, int N_slices, int N_images);
void cuda_collapse_responsabilities(real *d_respons, int N_slices, int N_images);
void cuda_calculate_best_rotation(real *d_respons, real* d_best_respons, int *d_best_rotation, int N_images, int N_slices);
real cuda_total_respons(real * d_respons, real * respons,int n);
void cuda_respons_max_expf(real* d_respons, real* max, int N_images, int allocate_slices, real* d_sum);
void cuda_norm_respons_sumexpf(real * d_respons,  real* d_sum, real* max, int N_images, int allocate_slices);
#endif

