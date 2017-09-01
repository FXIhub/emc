/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/

#include <emc_cuda_weight.h>
__global__ void calculate_weight_map_inner_kernel(real *weight_map, real width, real falloff) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int y = blockIdx.x;
    const int image_side = gridDim.x;

    real radius;
    real dx, dy;
    dy = (y - image_side/2) + 0.5;
    for (int x = tid; x < image_side; x += step) {
        dx = (x - image_side/2) + 0.5;
        // find distance to top left
        radius = sqrt(pow(dx, 2) + pow(dy, 2));
        // calculate logistic falloff
        weight_map[y*image_side + x] = 1. / (1. + exp((radius - width) * 5. / falloff));
    }
}

void cuda_calculate_weight_map_inner(real *d_weight_map, int image_side, real width, real falloff) {
    int nthreads = 256;
    int nblocks = image_side;
    calculate_weight_map_inner_kernel<<<nblocks,nthreads>>>(d_weight_map, width, falloff);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_calculate_weight_map: copy): %s\n",cudaGetErrorString(status));
    }
}
__global__ void calculate_weight_map_ring_kernel(real *weight_map, real inner_rad, real inner_falloff, real outer_rad, real outer_falloff) {
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int y = blockIdx.x;
    const int image_side = gridDim.x;

    real radius;
    real dx, dy;
    dy = (y - image_side/2) + 0.5;
    for (int x = tid; x < image_side; x+= step) {
        dx = (x - image_side/2) + 0.5;
        radius = sqrt(pow(dx, 2) + pow(dy, 2));
        weight_map[y*image_side + x] = (1. / (1. + exp(-(radius - inner_rad) * 5. / inner_falloff))) * (1. / (1. + exp((radius - outer_rad) * 5. / outer_falloff)));
    }
}

void cuda_calculate_weight_map_ring(real *d_weight_map, int image_side, real inner_rad, real inner_falloff, real outer_rad, real outer_falloff) {
    int nthreads = 256;
    int nblocks = image_side;
    calculate_weight_map_ring_kernel<<<nblocks, nthreads>>>(d_weight_map, inner_rad, inner_falloff, outer_rad, outer_falloff);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_calculate_weight_map: copy): %s\n",cudaGetErrorString(status));
    }
}
