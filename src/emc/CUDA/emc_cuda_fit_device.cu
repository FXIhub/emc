/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_fit.h>

__device__
void cuda_calculate_fit_error(float* slice, float *image,
                              int *mask, real scaling,
                              int N_2d, int tid, int step,
                              real *sum_cache, int *count_cache) {
    real sum = 0.0;
    int count = 0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            sum += abs((slice[i] - image[i]/scaling) / (slice[i] + image[i]/scaling));
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}

__device__
void cuda_calculate_fit2_error(float* slice, float *image,
                               int *mask, real scaling,
                               int N_2d, int tid, int step,
                               real *nom_cache, real *den_cache) {
    real nom = 0.0;
    real den = 0.0;
    const int i_max = N_2d;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            nom += abs(slice[i] - image[i]/scaling);
            den += abs(slice[i] + image[i]/scaling);
        }
    }
    nom_cache[tid] = nom;
    den_cache[tid] = den;
}

__device__
real cuda_calculate_single_fit_error(float* slice, float *image,
                                     int *mask, real scaling,
                                     int N_2d, int tid, int step) {
    __shared__ real sum_cache[256];
    __shared__ int count_cache[256];
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            sum += abs((slice[i] - image[i]/scaling) / (slice[i] + image[i]/scaling));
            count++;
        }
    }
   // __syncthreads();
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    inblock_reduce(sum_cache);
    inblock_reduce(count_cache);
    __syncthreads();
    return sum_cache[0]/count_cache[0];
}

__device__
real cuda_calculate_single_fit2_error(float* slice, float *image,
                                      int *mask, real scaling,
                                      int N_2d, int tid, int step) {
    __shared__ real nom_cache[256];
    __shared__ real den_cache[256];
    real nom = 0.0;
    real den = 0.0;
    const int i_max = N_2d;
    for (int i = tid; i < i_max; i+=step) {
        //if (mask[i] != 0 && slice[i] >= 0.0f) {
        if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
            nom += abs(slice[i] - image[i]/scaling);
            den += abs(slice[i] + image[i]/scaling);
        }
    }
    nom_cache[tid] = nom;
    den_cache[tid] = den;
    inblock_reduce(nom_cache);
    inblock_reduce(den_cache);

    return nom_cache[0]/(den_cache[0]);
}

__global__
void calculate_fit_kernel(real *slices, real *images, int *mask,
                          real *respons, real *fit, real sigma,
                          real *scaling, int N_2d, int slice_start){
    __shared__ real sum_cache[256];
    __shared__ int count_cache[256];
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    cuda_calculate_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
            scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step, sum_cache, count_cache);
    __syncthreads();
    inblock_reduce(sum_cache);
    inblock_reduce(count_cache);
    if (tid == 0) {
        if(count_cache[0]>0)
        atomicAdd(&fit[i_image], sum_cache[0]*respons[(slice_start+i_slice)*N_images+i_image] /count_cache[0]);
    }
}

__global__
void calculate_fit2_kernel(real *slices, real *images, int *mask,
                           real *respons, real *fit, real sigma,
                           real *scaling, int N_2d, int slice_start){
    __shared__ real nom_cache[256];
    __shared__ real den_cache[256];
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    cuda_calculate_fit2_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
            scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step, nom_cache, den_cache);

    inblock_reduce(nom_cache);
    inblock_reduce(den_cache);
    __syncthreads();
    if (tid == 0) {
        atomicAdd(&fit[i_image], nom_cache[0]/(den_cache[0])*respons[(slice_start+i_slice)*N_images+i_image]);
    }
}

__global__
void calculate_fit_best_rot_kernel(real *slices, real *images, int *mask,
                                   int *best_rot, real *fit,
                                   real *scaling, int N_2d, int slice_start){
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    real this_fit = 0.0;
    if (best_rot[i_image] == (slice_start+i_slice)) {
        this_fit = cuda_calculate_single_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
                scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);
      __syncthreads();
        if (tid == 0) {
            fit[i_image] = this_fit;
        }
    }
}


__global__
void calculate_fit_best_rot_local_kernel(real *slices, real *images, int *mask,
                                   int *best_rot, real *fit,
                                   real *scaling, int N_2d, int slice_start, int slice_backup){
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    real this_fit = 0.0;
    if (best_rot[i_image] == (slice_start+i_slice + slice_backup)) {
        this_fit = cuda_calculate_single_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
                scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);
      __syncthreads();
        if (tid == 0) {
            fit[i_image] = this_fit;
        }
    }
}


__global__
void calculate_fit_best_rot2_kernel(real *slices, real *images, int *mask,
                                    int *best_rot, real *fit,
                                    real *scaling, int N_2d, int slice_start){
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    real this_fit;
    if (best_rot[i_image] == (slice_start+i_slice)) {
        this_fit = cuda_calculate_single_fit2_error(&slices[i_slice*N_2d], &images[i_image*N_2d], mask,
                scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);

        if (tid == 0) {
            fit[i_image] = this_fit;
        }
    }
}

/* calcualte the fit as a function of radius */
__global__
void calculate_radial_fit_kernel(real * slices , real * images, int * mask,
                                 real * respons, real * scaling, real * radial_fit,
                                 real * radial_fit_weight, real * radius,
                                 int N_2d,  int side,  int slice_start){
    __shared__ real sum_cache[256]; //256
    __shared__ real weight_cache[256];
    const int max_radius = side/2;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;
    real error;
    int rad;

    if (tid < max_radius) {
        sum_cache[tid] = 0.0;
        weight_cache[tid] = 0.0;
    }
    __syncthreads();
    real this_resp = respons[(slice_start+i_slice)*N_images+i_image];
    for (int i = tid; i < N_2d; i += step) {
        if (mask[i] != 0 && slices[i_slice*N_2d+i] > 0.0f) {
            error = fabs((slices[i_slice*N_2d+i] - images[i_image*N_2d+i]/scaling[(slice_start+i_slice)*N_images+i_image]) /
                    (slices[i_slice*N_2d+i] + images[i_image*N_2d+i]/scaling[(slice_start+i_slice)*N_images+i_image]));
            rad = (int)radius[i];
            if (rad < max_radius) {
                atomicAdd(&sum_cache[rad],error*this_resp);
                atomicAdd(&weight_cache[rad],this_resp);
            }
        }
    }
    __syncthreads();
    if (tid < max_radius) {
        atomicAdd(&radial_fit[tid],sum_cache[tid]);
        atomicAdd(&radial_fit_weight[tid],weight_cache[tid]);
    }
}
/*
#ifdef __cplusplus
}
#endif
*/


