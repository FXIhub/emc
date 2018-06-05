/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_CUDA_COMMON_H
#define EMC_CUDA_COMMON_H
#include <emc_cuda.h>

template <typename T> struct x_log_x
{
    __host__ __device__
    T operator()(const T& x) const {
        if(x > 0){
            return x * logf(x);
        }else{
            return 0;
        }
    }
};


template<typename T>
struct absolute_difference //: public unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x, const T &y) const
  {
    return fabs(x-y);
  }
};


template<typename T>
struct rel_difference //: public unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x, const T &y) const
  {
      if( x!=0)
        return fabs(x-y)/fabs(x);
      else
        return 0;
  }
};

template<typename T>
__device__ void inblock_reduce(T * data){
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            data[threadIdx.x] += data[threadIdx.x + s];
        }
        __syncthreads();
    }
}


template<typename T>
__device__ void inblock_reduce_y(T * data){
    __syncthreads();
    for(unsigned int s=blockDim.y/2; s>0; s>>=1){
        if (threadIdx.y < s){
            data[threadIdx.y] += data[threadIdx.y+s];
        }
        __syncthreads();
    }
}

template<typename T> __device__ void inblock_maximum(T * data){
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            if(data[threadIdx.x] < data[threadIdx.x + s]){
                data[threadIdx.x] = data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
}

template<typename T> __device__ void inblock_maximum_index(T * data, int *index) {
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s/=2){
        if (threadIdx.x < s && data[threadIdx.x] < data[threadIdx.x + s]) {
            data[threadIdx.x] = data[threadIdx.x + s];
            index[threadIdx.x] = index[threadIdx.x + s];
        }
        __syncthreads();
    }
}

__device__ inline void atomicFloatAdd(float *address, float val)
{
    int i_val = __float_as_int(val);
    int tmp0 = 0;
    int tmp1;

    while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
    {
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}

__global__ void cuda_calculate_max_vectors_kernel(real* respons, int N_images, int N_slices, real* d_maxr);
__global__ void cuda_vector_divide_kernel(real * nom, real * den, int n);

#endif
