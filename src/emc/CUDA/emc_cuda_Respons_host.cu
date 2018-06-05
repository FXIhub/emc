/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include "emc_cuda_Respons.h"
#include "emc_cuda_memory_host.h"


/* Now takes start slice and slice chunk. Also removed memcopy, done separetely later. */
void cuda_calculate_responsabilities(real * d_slices, real * d_images, int * d_mask, real *d_weight_map,
                                     real sigma, real * d_scaling, real * d_respons, real *d_weights,
                                     int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff)
{
    cudaEvent_t k_begin;
    cudaEvent_t k_end;
    cudaEventCreate(&k_begin);
    cudaEventCreate(&k_end);
    cudaEventRecord (k_begin,0);

    dim3 nblocks(N_images,slice_chunk);
    int nthreads = 256;
    calculate_responsabilities_kernel<<<nblocks,nthreads>>>(d_slices, d_images, d_mask, d_weight_map,
                                                            sigma, d_scaling, d_respons, d_weights,
                                                            N_2d, slice_start, diff);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (calc resp): %s\n",cudaGetErrorString(status));
    }

    cudaEventRecord(k_end,0);
    cudaEventSynchronize(k_end);
    real k_ms;
    cudaEventElapsedTime (&k_ms, k_begin, k_end);
    //printf("cuda calculate_responsabilities time = %fms\n",k_ms);
}

void cuda_calculate_responsabilities_sum(real * respons, real * d_respons, int N_slices,
                                         int N_images){
    cudaMemcpy(respons,d_respons,sizeof(real)*N_slices*N_images,cudaMemcpyDeviceToHost);
    real respons_sum = 0;
    for(int i = 0;i<N_slices*N_images;i++){
        respons_sum += respons[i];
    }
    //printf("respons_sum = %f\n",respons_sum);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (resp sum): %s\n",cudaGetErrorString(status));
    }
}


void cuda_normalize_responsabilities_single(real *d_respons, int N_slices, int N_images) {
    int nblocks = N_images;
    int nthreads = 256;

    cuda_normalize_responsabilities_single_kernel<<<nblocks, nthreads>>>(d_respons, N_slices, N_images);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA Error (norm resp): %s\n", cudaGetErrorString(status));
    }
}

void cuda_normalize_responsabilities(real * d_respons, int N_slices, int N_images){
    int nblocks = N_images;
    int nthreads = 256;
    cuda_normalize_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons, N_slices, N_images);
    /*
  int nblocks = N_slices;
  int nthreads = 256;
  cuda_normalize_responsabilities_uniform_kernel<<<nblocks,nthreads>>>(d_respons, N_slices, N_images);
  */
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
    }
}


void cuda_collapse_responsabilities(real *d_respons, int N_slices, int N_images) {
    int nblocks = N_images;
    int nthreads = 256;
    collapse_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons, N_slices);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
    }
}


real cuda_total_respons(real * d_respons,int n){
    thrust::device_ptr<real> p(d_respons);
    x_log_x<real> unary_op;
    thrust::plus<real> binary_op;
    real init = 0;
    // Calculates sum_0^n d_respons*log(d_respons)
    return thrust::transform_reduce(p, p+n, unary_op, init, binary_op);
}



void cuda_calculate_best_rotation(real *d_respons, real* d_best_respons, int *d_best_rotation, int N_images, int N_slices){
    int nblocks = N_images;
    int nthreads = 256;
    calculate_best_rotation_kernel<<<nblocks, nthreads>>>(d_respons, d_best_respons, d_best_rotation, N_slices);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA Error (best rotation): %s\n", cudaGetErrorString(status));
    }
}
void cuda_respons_max_expf(real* d_respons, real* max, int N_images, int allocate_slices, real* d_sum){
    int nblocks = N_images;
    int nthreads = TNUM;
    //cuda_respons_max_expf_kernel<<<nblocks,nthreads>>>(d_respons, max, allocate_slices, N_images,d_sum);
    real * d_tmp;
    cuda_allocate_real(&d_tmp,allocate_slices*N_images);
    cuda_respons_max_expf_kernel<<<nblocks,nthreads>>>(d_respons,d_tmp, max, allocate_slices, N_images,d_sum);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda max expf): %s\n",cudaGetErrorString(status));
    }
    cuda_mem_free(d_tmp);
}



void cuda_norm_respons_sumexpf(real * d_respons,  real* d_sum, real* max, int N_images, int allocate_slices)
{
    int nblocks = N_images;
    int nthreads = TNUM;
    cuda_norm_respons_sumexpf_kernel <<<nblocks,nthreads>>>(  d_respons,  d_sum, max,  N_images, allocate_slices);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (max cuda_norm_respons_sumexpf): %s\n",cudaGetErrorString(status));
    }
}

