/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_2DPATTERNS_HOST_H
#define EMC_CUDA_2DPATTERNS_HOST_H
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_apply_masks(real *const d_images, const int *const d_masks, const int N_2d, const int N_images);

void cuda_apply_single_mask(real *const d_images, const int *const d_mask, const int N_2d, const int N_images);
/*
#ifdef __cplusplus
}
#endif
*/
#endif
