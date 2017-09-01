#ifndef EMC_CUDA_FIT_HOST_H
#define EMC_CUDA_FIT_HOST_H

void cuda_calculate_fit(real * slices, real * d_images, int * d_mask,
                        real * d_scaling, real * d_respons, real * d_fit, real sigma,
                        int N_2d, int N_images, int slice_start, int slice_chunk);
/*void cuda_calculate_fit_local(real * slices, real * d_images, int * d_mask,
                              real * d_scaling, real * d_respons, real * d_fit, real sigma,
                              int N_2d, int N_images, int slice_start, int slice_chunk,
                              real* nom, real* den);*/

void cuda_calculate_fit_best_rot(real *slices, real * d_images, int *d_mask,
                                 real *d_scaling, int *d_best_rot, real *d_fit,
                                 int N_2d, int N_images, int slice_start, int slice_chunk);
void cuda_calculate_fit_best_rot_local(real *slices, real * d_images, int *d_mask,
                                       real *d_scaling, int *d_best_rot, real *d_fit,
                                       int N_2d, int N_images, int slice_start, int slice_chunk, int offset);

void cuda_calculate_radial_fit(real *slices, real *d_images, int *d_mask,
                               real *d_scaling, real *d_respons, real *d_radial_fit,
                               real *d_radial_fit_weight, real *d_radius,
                               int N_2d, int side, int N_images, int slice_start,
                               int slice_chunk);

#endif
