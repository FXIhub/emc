/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */
#ifndef EMC_CUDA_MEMORY_H
#define EMC_CUDA_MEMORY_H

void cuda_allocate_slices(real ** slices, int side, int N_slices);
void cuda_allocate_model(real ** d_model, sp_3matrix * model);
void cuda_allocate_mask(int ** d_mask, sp_imatrix * mask);
void cuda_allocate_rotations(real ** d_rotations, Quaternion *rotations, int N_slices);
void cuda_allocate_rotations_chunk(real ** d_rotations, Quaternion * rotations, int start, int end);
void cuda_allocate_images(real ** d_images, sp_matrix ** images,  int N_images);
void cuda_allocate_masks(int ** d_images, sp_imatrix ** images,  int N_images);
void cuda_reset_model(sp_3matrix * model, real * d_model);
void cuda_copy_model(sp_3matrix * model, real *d_model);
void cuda_set_to_zero(real * x, int n);
void cuda_copy_real_to_device(real *x, real *d_x, int n);
void cuda_copy_real_to_host(real *x, real *d_x, int n);
void cuda_copy_int_to_device(int *x, int *d_x, int n);
void cuda_copy_int_to_host(int *x, int *d_x, int n);
void cuda_allocate_scaling(real ** d_scaling, int N_images);
void cuda_allocate_scaling_full(real **d_scaling, int N_images, int N_slices);
void cuda_copy_slice_chunk_to_host(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d);
void cuda_copy_slice_chunk_to_device(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d);
void cuda_allocate_real(real ** x, int n);
void cuda_allocate_int(int ** x, int n);
void cuda_allocate_coords(real ** d_x, real ** d_y, real ** d_z, sp_matrix * x,
                          sp_matrix * y, sp_matrix * z);

void cuda_allocate_weight_map(real **, int );
void cuda_set_real_array(real **d_array, int n, real value);

void cuda_copy_weight_to_device(real *x, real *d_x, int n, int taskid);
void cuda_reset_real(real *d_real, int len);
void cuda_copy_real(real *dst, real *src, int n);
void cuda_mem_free(real * d);
void cuda_copy_model_2_device (real ** d_model, sp_3matrix * model);

void cuda_copy_rotations_chunk(real ** d_rotations, Quaternion * rotations, int start, int end);

#endif
