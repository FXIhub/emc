/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include <emc_cuda_Respons.h>
/*#ifdef __cplusplus 
extern "C" {
#endif
*/
/* This responsability does not yet take scaling of patterns into accoutnt. */
__device__
void cuda_calculate_responsability_absolute(float *slice, float *image, int *mask, real *weight_map,
                                            real sigma, real scaling, int N_2d, int tid, int step,
                                            real * sum_cache, real *count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    real count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f) {
            sum += pow(slice[i] - image[i]/scaling,2)*weight_map[i];
            count += weight_map[i];
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__
void cuda_calculate_responsability_relative(float *slice, float *image, int *mask, real *weight_map,
                                            real sigma, real scaling, int N_2d, int tid, int step,
                                            real *sum_cache, real *count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    real count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.f) {
            sum += pow((slice[i] - image[i]/scaling) / slice[i], 2)*weight_map[i];
            count += weight_map[i];
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}


/* This responsability does not yet take scaling of patterns into accoutnt. */
__device__
void cuda_calculate_responsability_poisson(float *slice, float *image, int *mask, real *weight_map,
                                           real sigma, real scaling, int N_2d, int tid, int step,
                                           real * sum_cache, real * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    real count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] > 0.0f) {
            //sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+0.02), 2)*weight_map[i]; // 0.2 worked. this was used latest
            sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]), 2)*weight_map[i];
            //sum += (log(sqrt(slice[i])) + pow((slice[i] - image[i]/scaling) / sqrt(slice[i]), 2))*weight_map[i];
            count += weight_map[i];
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = 1;//count;
    //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__
void cuda_calculate_responsability_true_poisson (float *slice, float *image, int *mask, real *weight_map,
                                                 real sigma, real scaling, int N_2d, int tid, int step,
                                                 real * sum_cache, real * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    real count = 0;
    for (int i = tid; i < i_max; i+=step) {
      //if ( slice[i]>0.0f && mask[i] != 0 && image[i]>=0.0f  && scaling>0.0f){ //should have the condition on
        real phiW = scaling*slice[i];
        if (  mask[i] != 0  && phiW > 1e-10){ //should have the condition on
  
            //real low = slice[i]>0? logf(slice[i]):-1000;
            //real lop = scaling>0?logf(scaling):-1000;   
            //real low = slice[i]>0? logf(slice[i]):0;
            //real lop = scaling>0?logf(scaling):0;
            //{\displaystyle \ln n!\approx n\ln n-n+{\tfrac {1}{6}}\ln(8n^{3}+4n^{2}+n+{\tfrac {1}{30}})+{\tfrac {1}{2}}\ln \pi }.
            /*real loi = image[i] * logf(image[i])  -image[i] + 0.5*logf(2*3.141592653*image[i]) +
                                   1/(12*image[i]) -1/(360*pow(image[i],3)) +1/(1260*pow(image[i],5) -
                                   1/(1680*pow(image[i],7)));*/
            //sum += ( image[i]*low +  image[i]*lop - scaling*slice[i] - loi );//*weight_map[i];
            //phiW = phiW >1e-1? phiW:1e-1;

            sum += floor(image[i])*logf(phiW) - phiW;
            count += 1; 
      }
    }
    sum_cache[tid] = sum;
    count_cache[tid] =count;
}


__device__
void cuda_calculate_responsability_annealing_poisson(float *slice, float *image, int *mask, real* weight_map,
                                                     real sigma, real scaling, int N_2d, int tid,
                                                     int step, real *sum_cache, real *count_cache)
{
    real sum = 0.;
    const int i_max = N_2d;
    real count = 0;
    for (int i = tid; i < i_max; i += step) {
        if (mask[i] != 0 && slice[i] > 0.0f) {
            sum += logf(sqrt(scaling*slice[i])) + pow((scaling*slice[i] - image[i]), 2) / (scaling*slice[i])*weight_map[i];
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}

__device__
void cuda_calculate_responsability_absolute_atomic(float *slice,
                                                   float *image, int *mask, real sigma,
                                                   real scaling, int N_2d, int tid, int step,
                                                   real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] >= 0.0f) {
            sum += pow(slice[i] - image[i]/scaling,2);
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}

__device__
void cuda_calculate_responsability_poisson_atomic(float *slice, float *image,
                                                  int *mask, real sigma, real scaling,
                                                  int N_2d, int tid, int step,
                                                  real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] >= 0.0f) {
            sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]+0.02), 2); // 0.2 worked
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}

__device__
void cuda_calculate_responsability_true_poisson_atomic(float *slice, float *image,
                                                       int *mask, real sigma, real scaling,
                                                       int N_2d, int tid, int step,
                                                       real * sum_cache, int * count_cache)
{
    real sum = 0.0;
    const int i_max = N_2d;
    int count = 0;
    for (int i = tid; i < i_max; i+=step) {
        if (mask[i] != 0 && slice[i] >= 0.0f) {
            sum += image[i]*logf(slice[i]) +image[i]*logf(scaling) - scaling*slice[i];
            count++;
        }
    }
    sum_cache[tid] = sum;
    count_cache[tid] = count;
}
__global__
void calculate_responsabilities_kernel(float * slices, float * images, int * mask, real *weight_map,
                                       real sigma, real * scaling, real * respons, real *weights,
                                       int N_2d, int slice_start, enum diff_type diff)
{
    __shared__ real sum_cache[256];
    __shared__ real count_cache[256];
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int i_slice = blockIdx.y;
    int N_images = gridDim.x;

    if (diff == relative) {
        cuda_calculate_responsability_relative(&slices[i_slice*N_2d], &images[i_image*N_2d], mask, weight_map,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
                sum_cache, count_cache);
    } else if (diff == poisson) {
        cuda_calculate_responsability_poisson(&slices[i_slice*N_2d], &images[i_image*N_2d], mask, weight_map,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
                sum_cache, count_cache);
    } else if (diff == absolute) {
        /* This one was used for best result so far.*/
        cuda_calculate_responsability_absolute(&slices[i_slice*N_2d], &images[i_image*N_2d], mask, weight_map,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
                sum_cache, count_cache);
    } else if (diff == annealing_poisson) {
        cuda_calculate_responsability_annealing_poisson(&slices[i_slice*N_2d], &images[i_image*N_2d], mask, weight_map,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid,
                step, sum_cache, count_cache);
    } else if (diff == true_poisson){
        cuda_calculate_responsability_true_poisson(&slices[i_slice*N_2d], &images[i_image*N_2d], mask, weight_map,
                sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
                sum_cache, count_cache);
    }
    else{
        printf("Undefined noise model, please choose from relative, poisson, absolute, annealing_poisson, and true_poisson.\n");
        return;
    }
    inblock_reduce(sum_cache);
    inblock_reduce(count_cache);
    if(tid == 0 ){
        if ( diff == true_poisson)
            respons[(slice_start+i_slice)*N_images+i_image] = sum_cache[0]/weight_map[0];
        else
            respons[(slice_start+i_slice)*N_images+i_image] = -sum_cache[0]/2.0/count_cache[0]/pow(sigma,2);
    }

}
__global__ void calculate_best_rotation_kernel(real *respons,real* best_respons, int *best_rotation, int N_slices) {
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int N_images = gridDim.x;

    __shared__ real max_resp[256];
    __shared__ int max_index[256];
    max_resp[tid] =  -1;
    max_index[tid] = -1;
    real this_resp;
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        this_resp = respons[i_slice*N_images+i_image];
        if (this_resp > max_resp[tid] && this_resp > 1e-5) {
            max_resp[tid] = this_resp;
            max_index[tid] = i_slice;
        }
    }
    //__syncthreads();
    inblock_maximum_index(max_resp, max_index);
    __syncthreads();
    if (tid == 0) {
        best_rotation[i_image] = max_index[0];
        best_respons[i_image] = max_resp[0];
    }
}

__global__
void cuda_normalize_responsabilities_single_kernel(real *respons, int N_slices, int N_images) {
    __shared__ real max_cache[256];
    __shared__ int index_cache[256];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real this_resp;
    for (int i_slice= tid; i_slice < N_slices; i_slice += step) {
        this_resp = respons[i_slice*N_images+i_image];
        if (this_resp > max_cache[tid]) {
            max_cache[tid] = this_resp;
            index_cache[tid] = i_image;
        }
    }
    inblock_maximum_index(max_cache, index_cache);

    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        respons[i_slice*N_images+i_image] = 0.;
    }
    __syncthreads();
    if (tid == 0) {
        respons[index_cache[0]*N_images + i_image] = 1.;
    }
}

__global__
void cuda_normalize_responsabilities_uniform_kernel(real * respons, int N_slices, int N_images){
    __shared__ real cache[256];
    /* enforce uniform orientations first */
    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_image = tid; i_image < N_images; i_image += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    inblock_maximum(cache);
    real max_resp = cache[0];
    __syncthreads();
    for (int i_image = tid; i_image < N_images; i_image+= step) {
        respons[i_slice*N_images+i_image] -= max_resp;
    }

    cache[tid] = 0;
    for (int i_image = tid; i_image < N_images; i_image+=step) {
        if (respons[i_slice*N_images+i_image] > min_resp) {
            respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    inblock_reduce(cache);
    real sum = cache[0];
    __syncthreads();
    for (int i_image = tid; i_image < N_images; i_image+=step) {
        respons[i_slice*N_images+i_image] /= sum;
    }
    const real scaling_factor = ((float) N_images) / ((float) N_slices);
    for (int i_image = tid; i_image < N_images; i_image += step) {
        respons[i_slice*N_images+i_image] *= scaling_factor;
    }
}

__global__
void cuda_normalize_responsabilities_kernel(real * respons, int N_slices, int N_images){
    __shared__ real cache[256];

    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = -1.0e10f;
    for(int i_slice = tid;i_slice < N_slices;i_slice += step){
        if(cache[tid] < respons[i_slice*N_images+i_image]){
            cache[tid] = respons[i_slice*N_images+i_image];
        }
    }
    __syncthreads();
    inblock_maximum(cache);
    real max_resp = cache[0];
    __syncthreads();
    for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
        respons[i_slice*N_images+i_image] -= max_resp;
    }

    cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        if (respons[i_slice*N_images+i_image] > min_resp) {
            respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
        } else {
            respons[i_slice*N_images+i_image] = 0.0f;
        }
    }
    __syncthreads();
    inblock_reduce(cache);
    real sum = cache[0];
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        respons[i_slice*N_images+i_image] /= sum;
    }
}
__global__
void collapse_responsabilities_kernel(real *respons, int N_slices) {
    int i_image = blockIdx.x;
    int N_images = gridDim.x;
    int step = blockDim.x;
    int tid = threadIdx.x;

    real this_resp;
    real best_resp = 0.;
    int best_resp_index = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        this_resp = respons[i_slice*N_images + i_image];
        if (this_resp > best_resp) {
            best_resp = this_resp;
            best_resp_index = i_slice;
        }
    }

    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
        respons[i_slice*N_images + i_image] = 0.;
    }
    respons[best_resp_index*N_images + i_image] = 1.;
}

__global__ void cuda_respons_max_expf_kernel(real* respons,real* d_tmp,real* max,int N_slices,int N_images, real* d_sum){
    __shared__ real cache[256];
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    cache[tid] = 0;

    for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
        respons[i_slice*N_images+i_image] = respons[i_slice*N_images+i_image] - max[i_image];
    }
    for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
        //if (respons[i_slice*N_images+i_image] > min_resp) {
            respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
            cache[tid] += respons[i_slice*N_images+i_image];
       // }
       // else {
       //  respons[i_slice*N_images+i_image] = 0.0f;
      // }
    }
    __syncthreads();
    inblock_reduce(cache);
    d_sum[i_image] = cache[0];
}

__global__ void cuda_norm_respons_sumexpf_kernel(real * respons,  real* d_sum, real* max, int N_images, int allocate_slices){
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    for (int i_slice = tid; i_slice < allocate_slices; i_slice += step) {
        //real tmp = expf(respons[i_slice*N_images + i_image] -d_sum[i_image]);
        //if(  tmp> -1.0e10f)
        respons [i_slice*N_images + i_image] =  respons [i_slice*N_images + i_image] / d_sum[i_image];
        //else
        //respons [i_slice*N_images + i_image] =0.0f;
    }
}


