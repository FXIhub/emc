/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/

#ifndef EMC_CUDA_3DMODEL_HOST_H
#define EMC_CUDA_3DMODEL_HOST_H
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
real cuda_model_max(real * model, int model_size);
real cuda_model_average(real * model, int model_size);
void cuda_divide_model_by_weight(sp_3matrix * model, real * d_model, real * d_weight);
void cuda_normalize_model(sp_3matrix *model, real *d_model);
void cuda_output_device_model(real *d_model, char *filename, int side);
void cuda_blur_model(real *d_model, const int model_side, const real sigma);
/*
#ifdef __cplusplus
}
#endif
*/    
    
#endif
