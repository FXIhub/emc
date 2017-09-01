/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include <emc_cuda_Scaling.h>
void cuda_update_scaling(real * d_images, int * d_mask, real * d_scaling,
                         real *d_weighted_power, int N_images, int N_slices,
                         int N_2d, real * scaling)
{
    cudaEvent_t begin;
    cudaEvent_t end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord (begin,0);
    int nblocks = N_images;
    int nthreads = 256;
    cudaEvent_t k_begin;
    cudaEvent_t k_end;
    cudaEventCreate(&k_begin);
    cudaEventCreate(&k_end);
    cudaEventRecord (k_begin,0);
    slice_weighting_kernel<<<nblocks,nthreads>>>(d_images,d_mask,d_scaling,
                                                 d_weighted_power,N_slices,N_2d);
    cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);
    cudaEventRecord(k_end,0);
    cudaEventSynchronize(k_end);
    real k_ms;
    cudaEventElapsedTime (&k_ms, k_begin, k_end);

    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (update scaling): %s\n",cudaGetErrorString(status));
    }
    cudaEventRecord(end,0);
    cudaEventSynchronize (end);
    real ms;
    cudaEventElapsedTime (&ms, begin, end);
}


void cuda_update_weighted_power(real * d_images, real * d_slices, int * d_mask,
                                real * d_respons, real * d_weighted_power,
                                int N_images, int slice_start, int slice_chunk,
                                int N_2d)
{
    cudaEvent_t k_begin;
    cudaEvent_t k_end;
    cudaEventCreate(&k_begin);
    cudaEventCreate(&k_end);
    cudaEventRecord (k_begin,0);
    int nblocks = N_images;
    int nthreads = 256;

    calculate_weighted_power_kernel<<<nblocks,nthreads>>>(d_images,d_slices,d_mask,
                                                          d_respons,d_weighted_power, N_images,
                                                          slice_start,slice_chunk,N_2d);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error: %s\n",cudaGetErrorString(status));
    }

    cudaEventRecord(k_end,0);
    cudaEventSynchronize(k_end);
    real k_ms;
    cudaEventElapsedTime (&k_ms, k_begin, k_end);
}


///todo need to change to distributed version
void cuda_update_scaling_best(real *d_images, int *d_mask,  real *d_model,
                              real *d_scaling, real *d_weight_map,
                              real *d_respons, real *d_rotations,
                              real *x_coordinates, real *y_coordinates,
                              real *z_coordinates, int N_images, int N_slices,
                              int side, real *scaling) {
    //int nblocks = N_images;
    //int nthreads = 256;
    //const int N_2d = side*side;
    /*int *d_best_rotation;
    cudaMalloc(&d_best_rotation, N_images*sizeof(int));
    real *d_best_respons;
    cudaMalloc(&d_best_respons, N_images*sizeof(real));
    calculate_best_rotation_kernel<<<nblocks, nthreads>>>(d_respons,d_best_respons, d_best_rotation, N_slices);
    nthreads = 256;
    nblocks = N_images;
    update_scaling_best_kernel<<<nblocks,nthreads,N_2d*sizeof(real)>>>(d_scaling, d_best_rotation,d_images, d_model, d_mask, d_weight_map, d_rotations, x_coordinates, y_coordinates, z_coordinates, side, d_best_rotation);
    cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);*/
}

void cuda_update_scaling_full(real *d_images, real *d_slices, int *d_mask,
                              real *d_scaling, real *d_weight_map, int N_2d,
                              int N_images, int slice_start, int slice_chunk,
                              enum diff_type diff) {
    dim3 nblocks(N_images,slice_chunk);
    int nthreads = 256;
    update_scaling_full_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_scaling, d_weight_map, N_2d, slice_start, diff);
}
/*
#ifdef __cplusplus
}
#endif
*/
