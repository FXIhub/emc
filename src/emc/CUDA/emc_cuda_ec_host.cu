/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_ec.h>
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_test_interpolate() {
    printf("test interpolation start\n");
    int side = 10;
    real *model;
    cudaMalloc(&model, side*side*side*sizeof(real));
    real *d_return_value;
    cudaMalloc(&d_return_value, 1*sizeof(real));
    cuda_test_interpolate_kernel<<<1,1>>>(model, side, d_return_value);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (test interpolate): %s\n",cudaGetErrorString(status));
    }
    real *return_value = (real *)malloc(1*sizeof(real));
    cudaMemcpy(return_value, d_return_value, 1*sizeof(real), cudaMemcpyDeviceToHost);
    printf("interpolation result = %g\n", return_value[0]);

    printf("test interpolation end\n");
}

/* function now takes a start slice and a number of slices to retrieve */
void cuda_get_slices(sp_3matrix * model, real * d_model, real * d_slices, real * d_rot,
                     real * d_x_coordinates, real * d_y_coordinates,
                     real * d_z_coordinates, int start_slice, int slice_chunk){
    cudaEvent_t k_begin;
    cudaEvent_t k_end;
    cudaEventCreate(&k_begin);
    cudaEventCreate(&k_end);
    cudaEventRecord (k_begin,0);

    int rows = sp_3matrix_x(model);
    int cols = sp_3matrix_y(model);
    int N_2d = sp_3matrix_x(model)*sp_3matrix_y(model);
    int nblocks = slice_chunk;
    int nthreads = 256;
    get_slices_kernel<<<nblocks,nthreads>>>(d_model, d_slices, d_rot,d_x_coordinates,
                                            d_y_coordinates,d_z_coordinates,
                                            rows,cols,
                                            sp_3matrix_x(model),sp_3matrix_y(model),
                                            sp_3matrix_z(model), start_slice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (get slices): %s\n",cudaGetErrorString(status));
    }

    cudaEventRecord(k_end,0);
    cudaEventSynchronize(k_end);
    real k_ms;
    cudaEventElapsedTime (&k_ms, k_begin, k_end);
    //printf("cuda calculate slice time = %fms\n",k_ms);
}



void cuda_test_interpolate_set() {
    printf("test interpolation start\n");
    int side = 5;
    real *d_model;
    cudaMalloc(&d_model, side*side*side*sizeof(real));
    real *d_weight;
    cudaMalloc(&d_weight, side*side*side*sizeof(real));
    real *d_return_value;
    cudaMalloc(&d_return_value, 1*sizeof(real));
    cuda_test_interpolate_set_kernel<<<1,1>>>(d_model, d_weight, side);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (test interpolate): %s\n",cudaGetErrorString(status));
    }
    real *model = (real *) malloc(side*side*side*sizeof(real));
    cudaMemcpy(model, d_model, side*side*side*sizeof(real), cudaMemcpyDeviceToHost);
    real *weight = (real *) malloc(side*side*side*sizeof(real));
    cudaMemcpy(weight, d_weight, side*side*side*sizeof(real), cudaMemcpyDeviceToHost);

    printf("value\n");
    for (int z = 3; z <= 4; z++) {
        for (int y = 3; y <= 4; y++) {
            for (int x = 3; x <= 4; x++) {
                printf("%g ", model[side*side*z + side*y + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("test interpolation end\n");
}
/*
#ifdef __cplusplus
}
#endif
*/
