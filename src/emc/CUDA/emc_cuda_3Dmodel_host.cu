/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/

#include "emc_cuda_3Dmodel.h"
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
real cuda_model_max(real * model, int model_size){
    thrust::device_ptr<real> p(model);
    real max = thrust::reduce(p, p+model_size, real(0), thrust::maximum<real>());
    return max;
}

real cuda_model_diff(real* d_model,real * d_model_updated,int N_model){
    thrust::device_ptr<real> p(d_model);
    thrust::device_ptr<real> p1(d_model_updated);
    //thrust::transform(p, p + N_model, p1, p1,  absolute_difference<real>());
    thrust::transform(p, p + N_model, p1, p1,  rel_difference<real>());

    real sum_diff = thrust::reduce(p1, p1+N_model, real(0), thrust::plus<real>());

    return sum_diff;
}

real cuda_model_average(real * model, int model_size) {
    real *d_average;
    cudaMalloc(&d_average, sizeof(real));
    model_average_kernel<<<1,256>>>(model, model_size, d_average);
    real average;
    cudaMemcpy(&average, d_average, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_average);
    return average;
}
void cuda_divide_model_by_weight(sp_3matrix * model, real * d_model, real * d_weight){
    int n = sp_3matrix_size(model);
    int nthreads = 256;
    int nblocks = (n+nthreads-1)/nthreads;
    cuda_divide_model_kernel<<<nblocks,nthreads>>>(d_model,d_weight,n);
    cudaThreadSynchronize();
    cuda_mask_out_model_kernel<<<nblocks,nthreads>>>(d_model,d_weight,n);
}

void cuda_normalize_model(sp_3matrix *model, real *d_model) {
    int n = sp_3matrix_size(model);
    thrust::device_ptr<real> p(d_model);
    real model_average = cuda_model_average(d_model, sp_3matrix_size(model));
    printf("model average before normalization = %g\n", model_average);
    //real model_sum = thrust::reduce(p, p+n, real(0), thrust::plus<real>());
    //model_sum /= (real) n;
    thrust::transform(p, p+n,thrust::make_constant_iterator(1.0/model_average), p, thrust::multiplies<real>());
}


void cuda_normalize_model_given_mean(sp_3matrix *model, real *d_model, double mean) {
    int n = sp_3matrix_size(model);
    thrust::device_ptr<real> p(d_model);
    real model_average = cuda_model_average(d_model, sp_3matrix_size(model));
    printf("model average before normalization = %g\n", model_average);
    thrust::transform(p, p+n,thrust::make_constant_iterator(mean/model_average), p, thrust::multiplies<real>());

}

void cuda_output_device_model(real *d_model, char *filename, int side) {
    real *model = (real *)malloc(side*side*side*sizeof(real));
    cuda_copy_real_to_host(model, d_model, side*side*side);
    Image *model_out = sp_image_alloc(side, side, side);
    for (int i = 0; i < side*side*side; i++) {
        if (model[i] >= 0.) {
            model_out->image->data[i] = sp_cinit(model[i], 0.);
            model_out->mask->data[i] = 1;
        } else {
            //model_out->image->data[i] = sp_cinit(0., 0.);
            model_out->image->data[i] = sp_cinit(model[i], 0.);
            model_out->mask->data[i] = 0;
        }
    }
    sp_image_write(model_out, filename, 0);
    free(model);
    sp_image_free(model_out);
}

void cuda_blur_model(real *d_model, const int model_side, const real sigma) {
    cufftComplex *ft;
    cudaMalloc((void **)&ft, model_side*model_side*(model_side/2+1)*sizeof(cufftComplex));

    int *d_mask;
    cudaMalloc(&d_mask, model_side*model_side*model_side*sizeof(int));
    get_mask_from_model<<<(pow((double) model_side,3)/256+1), 256>>>(d_model, d_mask, pow((double) model_side, 3));

    cufftHandle plan;
    cufftPlan3d(&plan, model_side, model_side, model_side, CUFFT_R2C);
    cufftExecR2C(plan, d_model, ft);//, CUFFT_FORWARD);
    //multiply by gaussian kernel

    int nthreads = 256;
    dim3 nblocks(model_side, model_side);
    multiply_by_gaussian_kernel<<<nblocks,nthreads>>>(ft, sigma);
    cufftPlan3d(&plan, model_side, model_side, model_side, CUFFT_C2R);
    cufftExecC2R(plan, ft, d_model);//, CUFFT_INVERSE);
    apply_mask<<<(pow((double) model_side,3)/256+1),256>>>(d_model, d_mask, pow((double) model_side,3));

}
/*#ifdef __cplusplus
}
#endif
*/

