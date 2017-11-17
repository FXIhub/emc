#include <emc_cuda_Scaling.h>
/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
__global__
void slice_weighting_kernel(real * images,int * mask, real * scaling,
                            real *weighted_power, int N_slices, int N_2d)
{
    __shared__ real image_power[256];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int i_image = bid;
    image_power[tid] = 0.0;
    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] != 0) {
            image_power[tid] += pow(images[i_image*N_2d+i],2);
        }
    }
    inblock_reduce(image_power);
    if(tid == 0){
        scaling[i_image] = image_power[tid]/weighted_power[i_image];
    }
}

__global__
void calculate_weighted_power_kernel(real * images, real * slices, int * mask,
                                     real *respons, real * weighted_power,
                                     int N_images, int slice_start,
                                     int slice_chunk, int N_2d)
{
    __shared__ real correlation[256];
    int step = blockDim.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int i_image = bid;
    for (int i_slice = 0; i_slice < slice_chunk; i_slice++) {
        correlation[tid] = 0.0;
        for (int i = tid; i < N_2d; i+=step) {
            if (mask[i] != 0 && slices[i_slice*N_2d+i] > 0.0f) {
                correlation[tid] += images[i_image*N_2d+i]*slices[i_slice*N_2d+i];
            }
        }
        inblock_reduce(correlation);
        if(tid == 0){
            weighted_power[i_image] += respons[(slice_start+i_slice)*N_images+i_image]*correlation[tid];
        }
    }
}

__global__
void update_scaling_best_kernel(real *scaling, real *images, real *model,
                                int *mask, real *weight_map, real *rotations,
                                real *x_coordinates, real *y_coordinates,
                                real *z_coordinates, int side, int *best_rotation)
{
    int step = blockDim.x;
    int i_image = blockIdx.x;
    int tid = threadIdx.x;
    const int N_2d = side*side;
    extern __shared__ real this_slice[];
    cuda_get_slice(model, this_slice, &rotations[4*best_rotation[i_image]],
            x_coordinates, y_coordinates, z_coordinates,
            side, side, side, side, side, tid, step);
    real this_scaling = calculate_scaling_poisson(&images[N_2d*i_image], this_slice, mask, weight_map, N_2d, tid, step);
    if (tid == 0) {
        scaling[i_image] = this_scaling;
    }
}

__global__
void update_scaling_full_kernel(real *images, real *slices, int *mask,
                                real *scaling, real *weight_map, int N_2d,
                                int slice_start, enum diff_type diff)
{
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int i_image = blockIdx.x;
    const int i_slice = blockIdx.y;
    const int N_images = gridDim.x;
    real this_scaling;
    if (diff == poisson) {
        this_scaling = calculate_scaling_poisson(&images[N_2d*i_image], &slices[N_2d*i_slice], mask, weight_map, N_2d, tid, step);
    } else if (diff == absolute) {
        this_scaling = calculate_scaling_absolute(&images[N_2d*i_image], &slices[N_2d*i_slice], mask, weight_map, N_2d, tid, step);
    } else if (diff == relative) {
        this_scaling = calculate_scaling_relative(&images[N_2d*i_image], &slices[N_2d*i_slice], mask, weight_map, N_2d, tid, step);
    }
    else if(diff==true_poisson)
        this_scaling = calculate_scaling_true_poisson(&images[N_2d*i_image], &slices[N_2d*i_slice], mask, weight_map, N_2d, tid, step);

    __syncthreads();
    if (tid == 0) {
        scaling[(slice_start+i_slice)*N_images+i_image] = this_scaling;
    }
}
