#include<emc_cuda_Slice.h>
/*#ifdef __cplusplus 
extern "C" {
#endif
*/

__global__ void update_slices_kernel(real* images, real* slices, int* mask, real* respons,
                                     real* scaling, int* active_images, int N_images,
                                     int slice_start, int N_2d, real* slices_total_respons,
                                     real* rot, real* x_coord, real* y_coord, real* z_coord,
                                     real* model, int slice_rows, int slice_cols, int model_x,
                                     int model_y, int model_z){
    /* each block takes care of 1 slice */
    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = 0.0f;

    for (int i = tid; i < N_2d; i+=step) {
        if (mask[i] != 0) {
            real sum = 0;
            for (int i_image = 0; i_image < N_images; i_image++) {
                if (active_images[i_image] > 0) {
                    sum += images[i_image*N_2d+i]*
                            respons[(slice_start+i_slice)*N_images+i_image]/scaling[(slice_start+i_slice)*N_images+i_image];
                }
            }
            slices[i_slice*N_2d+i] = sum;
        } else {
            slices[i_slice*N_2d+i] =0;// -1.0;
        }
    }
    for (int i_image = 0; i_image < N_images; i_image++) {
        if (active_images[i_image] > 0) {
            total_respons += respons[(slice_start+i_slice)*N_images+i_image];
        }
    }
    if(tid == 0){
        slices_total_respons[i_slice] =  total_respons;
    }
    if(total_respons > 1e-10f){
        for (int i = tid; i < N_2d; i+=step) {
            if (mask[i] != 0) {
                slices[i_slice*N_2d+i] /= total_respons;
            }
        }
    }
}


__global__ void update_slices_true_poisson_kernel(real* images, real* slices, int* mask, real* respons,
                                                  real* scaling, int* active_images, int N_images,
                                                  int slice_start, int N_2d, real* slices_total_respons,
                                                  real* rot, real* x_coord, real* y_coord, real* z_coord,
                                                  real* model, int slice_rows, int slice_cols, int model_x,
                                                  int model_y, int model_z){

    int i_slice = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = 0.0f;
    //real cutoff = 100000;
    for (int i = tid; i < N_2d; i+=step) {
        real sum = 0.;
        real norm = 0;
        for (int i_image = 0; i_image < N_images; i_image++) {
            if (mask[i] != 0 && active_images[i_image] > 0) {
                //if(images[i_image*N_2d+i]<cutoff){
                sum += images[i_image*N_2d+i] *  respons[(slice_start+i_slice)*N_images+i_image];
                norm +=   respons[(slice_start+i_slice)*N_images+i_image] * scaling[(slice_start+i_slice)*N_images+i_image];
                //}
                //total_respons += respons[(slice_start+i_slice)*N_images+i_image];
            }
        }
        if (norm > 1e-10f) {
            slices[i_slice*N_2d+i] = (sum / norm) ;
        } else {
            slices[i_slice*N_2d+i] = -1.;
        }
    }

    for (int i_image = 0; i_image < N_images; i_image++) {
        if (active_images[i_image] > 0) {
            total_respons += respons[(slice_start+i_slice)*N_images+i_image] ;
        }
    }
    if(tid == 0){
        slices_total_respons[i_slice] =  total_respons;
    }
}

__global__
void update_slices_final_kernel(real* images, real* slices, int* mask, real* respons,
                                real* scaling, int* active_images, int N_images,
                                int slice_start, int N_2d,
                                real* slices_total_respons, real* rot,
                                real* x_coord, real* y_coord, real* z_coord,
                                real* model, real* weight,
                                int slice_rows, int slice_cols,
                                int model_x, int model_y, int model_z){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    real total_respons = 0.0f;
    int i_slice = bid;
    for (int i = tid; i < N_2d; i+=step) {
        real sum = 0.0;
        real count = 0.0;
        for (int i_image = 0; i_image < N_images; i_image++) {
            if (active_images[i_image] > 0 && images[i_image*N_2d+i] >= 0.0) {
                sum += images[i_image*N_2d+i]*
                        respons[(slice_start+i_slice)*N_images+i_image]/scaling[(slice_start+i_slice)*N_images+i_image];
                count += respons[(slice_start+i_slice)*N_images+i_image];
            }
        }
        if (count > 1.0e-10) {
            slices[i_slice*N_2d+i] = sum/count;
        } else {
            slices[i_slice*N_2d+i] = -1.0;
        }
    }
    for (int i_image = 0; i_image < N_images; i_image++) {
        if (active_images[i_image] > 0) {
            total_respons += respons[(slice_start+i_slice)*N_images+i_image];
        }
    }
    if(tid == 0){
        slices_total_respons[bid] =  total_respons;
    }
}
/*
#ifdef __cplusplus 
}
#endif
*/
