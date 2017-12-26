/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_EC_HOST_H
#define EMC_CUDA_EC_HOST_H
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_test_interpolate();
void cuda_get_slices(sp_3matrix * model, real * d_model, real * d_slices, real * d_rot,
                     real * d_x_coordinates, real * d_y_coordinates,
                     real * d_z_coordinates, int start_slice, int slice_chunk);
void cuda_test_interpolate_set();
void cuda_replace_slices(real*d_slices, real* d_average_slice,int*d_msk, int slice_chunk, int N_2d);

/*
#ifdef __cplusplus
}
#endif
*/
#endif
