/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */
#ifndef EMC_CUDA_SLICE_H
#define EMC_CUDA_SLICE_H
#include <emc_cuda.h>
#include <emc_cuda_ec.h>
/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
// DEVICE FUNCTIONS
// GLOBAL FUNCTIONS
__global__
void update_slices_kernel(real* images, real* slices, int* mask, real* respons,
                     real* scaling, int* active_images, int N_images,
                     int slice_start, int N_2d, real* slices_total_respons,
                     real* rot, real* x_coord, real* y_coord, real* z_coord,
                     real* model, int slice_rows, int slice_cols, int model_x,
                     int model_y, int model_z);

__global__
void update_slices_final_kernel(real* images, real* slices, int* mask, real* respons,
                                real* scaling, int* active_images, int N_images,
                                int slice_start, int N_2d,
                                real* slices_total_respons, real* rot,
                                real* x_coord, real* y_coord, real* z_coord,
                                real* model, real* weight,
                                int slice_rows, int slice_cols,
                                int model_x, int model_y, int model_z);

#endif

/*
#ifdef __cplusplus 
}
#endif
*/
