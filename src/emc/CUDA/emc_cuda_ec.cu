/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include <emc_cuda_ec.h>

/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
/* updated to use rotations with an offset start. */


__global__
void get_slices_kernel(real * model, real * slices, real *rot, real *x_coordinates,
                       real *y_coordinates, real *z_coordinates, int slice_rows,
                       int slice_cols, int model_x, int model_y, int model_z,
                       int start_slice){
    int bid = blockIdx.x;
    int i_slice = bid;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int N_2d = slice_rows*slice_cols;

    cuda_get_slice_interpolate(model,&slices[N_2d*i_slice],&rot[4*(start_slice+i_slice)],x_coordinates,
            y_coordinates,z_coordinates,slice_rows,slice_cols,model_x,model_y,
            model_z,tid,step);


}

__global__ 
void cuda_test_interpolate_kernel(real *model, int side, real *return_value) {
    printf("enter kernel %d %d\n", blockIdx.x, threadIdx.x);
    for (int x = 0; x < side; x++) {
        for (int y = 0; y < side; y++) {
            for (int z = 0; z < side; z++) {
                model[side*side*z + side*y + x] = (real)y + 0.5;
            }
        }
    }
    model[side*side*4 + side*4 + 4] = -1.;
    model[side*side*5 + side*4 + 4] = -1.;

    real interp_x = 4.5;
    real interp_y = 4.5;
    real interp_z = 4.5;
    return_value[0] = interpolate_model_get(model, side, side, side, interp_x, interp_y, interp_z);
    printf("interpolate at %g %g %g -> %g\n", interp_x, interp_y, interp_z, interpolate_model_get(model, side, side, side, interp_x, interp_y, interp_z));
}

__global__ void cuda_test_interpolate_set_kernel(real *model, real *weight, int side) {
    printf("enter kernel %d %d\n", blockIdx.x, threadIdx.x);
    for (int x = 0; x < side; x++) {
        for (int y = 0; y < side; y++) {
            for (int z = 0; z < side; z++) {
                model[side*side*z + side*y + x] = 0.;
                weight[side*side*z + side*y + x] = 0.;
            }
        }
    }
    real interp_x = 3.4;
    real interp_y = 4.6;
    real interp_z = 3.5;
    real value = 1.;
    real value_weight = 1.;
    interpolate_model_set(model, weight, side, side, side, interp_x, interp_y, interp_z, value, value_weight);
}

__global__
void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
                          real * scaling, int N_images, int N_2d,
                          real * slices_total_respons, real * rot,
                          real * x_coord, real * y_coord, real * z_coord,
                          real * model, real * weight,
                          int slice_rows, int slice_cols,
                          int model_x, int model_y, int model_z){
    /* each block takes care of 1 slice */
    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = slices_total_respons[i_slice];
    if (INTERPOLATION_METHOD ==0)
    {
        cuda_insert_slice_interpolate_NN(model, weight, &slices[i_slice*N_2d], mask, total_respons,
                &rot[4*i_slice], x_coord, y_coord, z_coord,
                slice_rows, slice_cols, model_x, model_y, model_z, tid, step);
    }
    else{
    if(total_respons > 1e-10f){
        cuda_insert_slice_interpolate(model, weight, &slices[i_slice*N_2d], mask, total_respons,
                &rot[4*i_slice], x_coord, y_coord, z_coord,
                slice_rows, slice_cols, model_x, model_y, model_z, tid, step);

    }
    }
}


__global__
void     replace_slices_kernel(real* d_slices, real* d_average_slice,int* d_msk,int slice_chunk, int N_2d){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slice_chunk*N_2d) {
        //d_slices[i] = d_slices[i]>0?d_slices[i]:d_average_slice[i%N_2d];
        d_slices[i] = d_msk[i%N_2d] ==0?-1:d_slices[i];

    }
}

/*
#ifdef __cplusplus 
}
#endif
*/
