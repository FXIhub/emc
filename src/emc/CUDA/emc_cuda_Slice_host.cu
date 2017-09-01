/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_Slice.h>
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_update_slices(real * d_images, real * d_slices, int * d_mask,
                        real * d_respons, real * d_scaling, int * d_active_images, int N_images,
                        int slice_start, int slice_chunk, int N_2d,
                        sp_3matrix * model, real * d_model,
                        real *d_x_coordinates, real *d_y_coordinates,
                        real *d_z_coordinates, real *d_rot,
                        real * d_weight, sp_matrix ** images){
    dim3 nblocks = slice_chunk;//N_slices;
    int nthreads = 256;
    real * d_slices_total_respons;
    cudaMalloc(&d_slices_total_respons,sizeof(real)*slice_chunk);
    cudaEvent_t k_begin;
    cudaEvent_t k_end;
    cudaEventCreate(&k_begin);
    cudaEventCreate(&k_end);
    cudaEventRecord (k_begin,0);

    update_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                               d_scaling, d_active_images, N_images, slice_start, N_2d,
                                               d_slices_total_respons, d_rot,d_x_coordinates,
                                               d_y_coordinates,d_z_coordinates,d_model,
                                               sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));
    cudaThreadSynchronize();
    insert_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                               d_scaling, N_images, N_2d,
                                               d_slices_total_respons, d_rot,d_x_coordinates,
                                               d_y_coordinates,d_z_coordinates,d_model, d_weight,
                                               sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));
    cudaEventRecord(k_end,0);
    cudaEventSynchronize(k_end);
    real k_ms;
    cudaEventElapsedTime (&k_ms, k_begin, k_end);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (update slices): %s\n",cudaGetErrorString(status));
    }
}

void cuda_update_slices_final(real * d_images, real * d_slices, int * d_mask,
                              real * d_respons, real * d_scaling, int * d_active_images,
                              int N_images, int slice_start, int slice_chunk, int N_2d,
                              sp_3matrix * model, real * d_model,
                              real *d_x_coordinates, real *d_y_coordinates,
                              real *d_z_coordinates, real *d_rot,
                              real * d_weight, sp_matrix ** images){
    dim3 nblocks = slice_chunk;//N_slices;
    int nthreads = 256;
    real * d_slices_total_respons;
    cudaMalloc(&d_slices_total_respons,sizeof(real)*slice_chunk);
    cudaEvent_t k_begin;
    cudaEvent_t k_end;
    cudaEventCreate(&k_begin);
    cudaEventCreate(&k_end);
    cudaEventRecord (k_begin,0);

    update_slices_final_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                                     d_scaling, d_active_images, N_images, slice_start, N_2d,
                                                     d_slices_total_respons, d_rot,d_x_coordinates,
                                                     d_y_coordinates,d_z_coordinates,d_model, d_weight,
                                                     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));

    cudaThreadSynchronize();
    insert_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
                                               d_scaling, N_images, N_2d,
                                               d_slices_total_respons, d_rot,d_x_coordinates,
                                               d_y_coordinates,d_z_coordinates,d_model, d_weight,
                                               sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
            sp_3matrix_x(model),sp_3matrix_y(model),
            sp_3matrix_z(model));
    cudaEventRecord(k_end,0);
    cudaEventSynchronize(k_end);
    real k_ms;
    cudaEventElapsedTime (&k_ms, k_begin, k_end);
    //printf("cuda kernel slice update time = %fms\n",k_ms);

    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (update slices): %s\n",cudaGetErrorString(status));
    }
}
/*
#ifdef __cplusplus
}
#endif
*/
