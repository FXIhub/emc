/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_fit.h>

void cuda_calculate_fit(real * slices, real * d_images, int * d_mask,
                        real * d_scaling, real * d_respons, real * d_fit, real sigma,
                        int N_2d, int N_images, int slice_start, int slice_chunk){
    //call the kernel
    dim3 nblocks(N_images,slice_chunk);
    int nthreads = 256;
    calculate_fit_kernel<<<nblocks,nthreads>>>(slices, d_images, d_mask,
                                               d_respons, d_fit, sigma, d_scaling,
                                               N_2d, slice_start);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
    }
}

void cuda_calculate_fit_best_rot(real *slices, real * d_images, int *d_mask,
                                 real *d_scaling, int *d_best_rot, real *d_fit,
                                 int N_2d, int N_images, int slice_start, int slice_chunk) {
    dim3 nblocks(N_images, slice_chunk);
    int nthreads = 256;
    calculate_fit_best_rot_kernel<<<nblocks, nthreads>>>(slices, d_images, d_mask,
                                                         d_best_rot, d_fit, d_scaling,
                                                         N_2d, slice_start);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
    }
}

void cuda_calculate_fit_best_rot_local(real *slices, real * d_images, int *d_mask,
                                        real *d_scaling, int *d_best_rot, real *d_fit,
                                        int N_2d, int N_images, int slice_start, int slice_chunk, int offset) {
    dim3 nblocks(N_images, slice_chunk);
    int nthreads = 256;
    calculate_fit_best_rot_local_kernel<<<nblocks, nthreads>>>(slices, d_images, d_mask,
                                                         d_best_rot, d_fit, d_scaling,
                                                         N_2d, slice_start, offset);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
    }
}


void cuda_calculate_radial_fit(real *slices, real *d_images, int *d_mask,
                               real *d_scaling, real *d_respons, real *d_radial_fit,
                               real *d_radial_fit_weight, real *d_radius,
                               int N_2d, int side, int N_images, int slice_start,
                               int slice_chunk){
    dim3 nblocks(N_images,slice_chunk);
    int nthreads = 256;
    calculate_radial_fit_kernel<<<nblocks,nthreads>>>(slices, d_images, d_mask,
                                                      d_respons, d_scaling, d_radial_fit,
                                                      d_radial_fit_weight, d_radius,
                                                      N_2d, side, slice_start);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess) {
        printf("CUDA Error (radial fit): %s\n",cudaGetErrorString(status));
    }
}
/*
#ifdef __cplusplus
}
#endif
*/
