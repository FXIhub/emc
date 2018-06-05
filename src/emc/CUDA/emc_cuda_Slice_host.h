/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */
#ifndef EMC_CUDA_SLICE_HOST_H
#define EMC_CUDA_SLICE_HOST_H
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_update_slices_final(real * d_images, real * d_slices, int * d_mask,
                              real * d_respons, real * d_scaling, int * d_active_images,
                              int N_images, int slice_start, int slice_chunk, int N_2d,
                              sp_3matrix * model, real * d_model,
                              real *d_x_coordinates, real *d_y_coordinates,
                              real *d_z_coordinates, real *d_rot,
                              real * d_weight, sp_matrix ** images);
void cuda_insert_slices(real * d_images, real * d_slices, int * d_mask,
                        real * d_respons, real * d_scaling,real*d_slice_total_respons, int * d_active_images,
                              int N_images, int slice_start, int slice_chunk, int N_2d,
                              sp_3matrix * model, real * d_model,
                              real *d_x_coordinates, real *d_y_coordinates,
                              real *d_z_coordinates, real *d_rot,
                              real * d_weight, sp_matrix ** images);
    //Mstep Update 2D slices
void cuda_update_slices(real * d_images, real * d_slices, int * d_mask,
                        real * d_respons, real * d_scaling, int * d_active_images,
                        int N_images, int slice_start, int slice_chunk, int N_2d,
                        sp_3matrix * model, real * d_model,
                        real *d_x_coordinates, real *d_y_coordinates,
                        real *d_z_coordinates, real *d_rot,
                        real * d_weight, sp_matrix ** images);
void cuda_update_true_poisson_slices(real * d_images, real * d_slices, int * d_mask,
                        real * d_respons, real * d_scaling, int * d_active_images,
                        int N_images, int slice_start, int slice_chunk, int N_2d,
                        sp_3matrix * model, real * d_model,
                        real *d_x_coordinates, real *d_y_coordinates,
                        real *d_z_coordinates, real *d_rot,
                        real * d_weight, sp_matrix ** images);
/*
#ifdef __cplusplus
}
#endif
*/
#endif






