#include <time.h>
#include <emc.h>
#include <hdf5.h>
#include <getopt.h>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <math.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>

using namespace std;
#define real float

struct LOG: public thrust::unary_function<real, real>
{
    __host__ __device__  real  operator()(real x)
    {
        return logf(x);
    }
};
struct isnan_test {
    __host__ __device__ bool operator()(const float a) const {
        return isnan(a);
    }
};


int main(int argc, char **argv)
{
    char configuration_filename[PATH_MAX] = "./emc_1GPU.conf";
    Configuration conf;
    int conf_return = read_configuration_file(configuration_filename, &conf);
    char filename_buffer[256];
    const int N_images =500;
    const int slice_chunk = 6000;
    const int N_2d = 64*64;
    const int N_model =64*64*64;
    const int N_slices = 6000;
    const int COUNTS = 6;
    const real sigma = 0.2;
    /* In this loop through the chunks the responsabilities are
       updated. */
    clock_t difference;
    int  msec = 0;
    //size_t m = 1000, n= 86520;
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasHandle_t hd;
    curandGenerator_t rng;
    cublasCreate(&hd);
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);



    thrust::device_vector<real> images(N_images*N_2d);
    real* d_images = thrust::raw_pointer_cast(&images[0]);
    curandGenerateUniform(rng, d_images, images.size());

    thrust::device_vector<real> respons(N_slices * N_images,0);
    thrust::device_vector<real> respons2(N_slices * N_images,0);

    thrust::device_vector<real> responstmp(N_slices* N_images);
    thrust::device_vector<real> slices(N_2d * slice_chunk);
    thrust::device_vector<real> scaling(N_slices * N_images);
    thrust::device_vector<int> mask(N_2d,1);
    thrust::device_vector<real> weight_map(N_2d,1);
    thrust::device_vector<real> weights(N_slices,1);

    real* d_slices = thrust::raw_pointer_cast(&slices[0]);
    real* d_scaling = thrust::raw_pointer_cast(&scaling[0]);
    real* d_respons = thrust::raw_pointer_cast(&respons[0]);
    real* d_respons2 = thrust::raw_pointer_cast(&respons2[0]);

    real* d_responstmp = thrust::raw_pointer_cast(&responstmp[0]);
    int * d_mask = thrust::raw_pointer_cast(&mask[0]); //all one
    real * d_weight_map = thrust::raw_pointer_cast(&weight_map[0]);
    real * d_weights = thrust::raw_pointer_cast(&weights[0]);

    curandGenerateUniform(rng, d_images, images.size());
    curandGenerateUniform(rng, d_slices, slices.size());
    curandGenerateUniform(rng, d_scaling, scaling.size());
    curandGenerateUniform(rng, d_weights, weights.size());
    curandGenerateUniform(rng, d_weight_map, weight_map.size());
/*   cout << "images" <<endl;
    for (int i =0; i<N_2d*N_images; i++)
        cout << images[i] <<" ";
    cout<<endl << "slices"<<endl;
    for (int i =0; i<N_2d*N_slices; i++)
        cout << slices[i] <<" ";
    cout<<endl << "scaling"<<endl;
    for (int i =0; i<N_slices * N_images; i++)
        cout << scaling[i] <<" ";
    cout <<endl<<endl;
*/
    thrust::device_vector<real> onesN2d(N_2d*N_images,1);
    thrust::device_vector<real> onesNslicechunck(N_2d*slice_chunk ,1);

    thrust::device_vector<real> sum_pix_of_slices(slice_chunk ,1);
    thrust::device_vector<real> sum_pix_of_image(N_images ,1);

    real* d_onesN2d = thrust::raw_pointer_cast(&onesN2d[0]);
    real* d_onesNslicechunck = thrust::raw_pointer_cast(&onesNslicechunck[0]);

    real* d_sum_pix_of_slices = thrust::raw_pointer_cast(&sum_pix_of_slices[0]);
    real* d_sum_pix_of_image = thrust::raw_pointer_cast(&sum_pix_of_image[0]);

    const real one = 1.0;
    const real mone = -1.0;
    const real zero = 0;

    int current_chunk =0;
    clock_t before = clock();
    // for (int count  = 0; count < COUNTS; count++){
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
        if (slice_start + slice_chunk >= N_slices) {
            current_chunk = N_slices - slice_start;
        } else {
            current_chunk = slice_chunk;
        }
        cuda_calculate_responsabilities(d_slices, d_images, d_mask, d_weight_map,
                                        sigma, d_scaling, d_respons2, d_weights,
                                        N_2d, N_images, slice_start,
                                        current_chunk, conf.diff);
    }
    difference = clock() - before;
    msec = difference * 1000 / CLOCKS_PER_SEC;
    cout <<" calc respons raw time = " << msec <<endl;
    //}
    for (int i = 2000; i<2100; i++) cout<< respons2[i] <<" ";
//    for (int i =0; i<N_slices * N_images; i++)
        //cout<< respons2[i] <<" ";

    cublasStatus_t st;

    before = clock();
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
        if (slice_start + slice_chunk >= N_slices) {
            current_chunk = N_slices - slice_start;
        } else {
            current_chunk = slice_chunk;
        }

        //sum_i \phi_jk * W_ij with sum_i W_{ij}

        //cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
        //const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)


        cublasSgemv(hd, CUBLAS_OP_T, N_2d, slice_chunk,
                    &one,&d_slices[slice_start], N_2d, d_onesNslicechunck, 1, &zero, d_sum_pix_of_slices, 1);

        //cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const real *A, int lda,
        //const real *x, int incx, real *C, int ldc)
        st = cublasSdgmm(hd, CUBLAS_SIDE_RIGHT//LEFT
	, slice_chunk, N_images, &d_scaling[slice_start],slice_chunk,
                         d_sum_pix_of_slices, 1, &d_respons[slice_start], slice_chunk);

        cout<<endl<<endl<<endl;
        cout <<endl<<endl<<endl;
        /*cout << "sum_i \phi_jk * W_ij with sum_i W_{ij} right "<<endl;
        for (int i =0; i< N_slices * N_images; i++)
            cout<< respons[i] <<" ";
*/
       //log phi_{jk} sum_i K_{ik}
        cublasSgemv(hd, CUBLAS_OP_T, N_2d, N_images,&one,d_images, N_2d, d_onesN2d, 1, &zero, d_sum_pix_of_image, 1);
        thrust::transform(scaling.begin(), scaling.end(), scaling.begin(), LOG());
          //cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const real *A, int lda,
            //const real *x, int incx, real *C, int ldc)
            cublasSdgmm(hd, CUBLAS_SIDE_RIGHT// must RIGHT
		, N_slices, N_images, d_scaling, N_slices,
                        d_sum_pix_of_image, 1, d_responstmp, N_slices);
            /*cout<<endl<<endl<<endl;
            cout <<endl<<endl<<endl;
            cout << "responstmp log phi_{jk} sum_i K_{ik} right"<<endl;
            for (int i =0; i<N_slices * N_images; i++)
                cout<< responstmp[i] <<" ";
*/
            //cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const real *alpha,
            //const real *A, int lda, const real *B, int ldb, const real *beta, real *C, int ldc) shrink down k
            //sum_i K logW(doing) - sum_i phi W(did)
         thrust::transform(slices.begin(), slices.end(), slices.begin(), LOG());

            //C = α op ( A ) op ( B ) + β C
            cublasSgemm(hd, CUBLAS_OP_T , CUBLAS_OP_N , N_slices, N_images,N_2d,&one ,
                        d_slices, N_2d, d_images,N_2d,  &mone, &d_respons[slice_start],N_slices );
            cout<<endl<<endl<<endl;
            cout <<endl<<endl<<endl;
            cout << "respons sum_i K logW - sum_i phi W"<<endl;
/*
            for (int i =0; i<N_slices * N_images; i++)
                cout<< respons[i] <<" ";
  */          // sum over
            thrust::transform(respons.begin(), respons.end(),
                              responstmp.begin(),respons.begin(),thrust::plus<float>());
            cout<<endl<<endl<<endl;
            cout <<endl<<endl<<endl;
            cout << "responssum"<<endl;
//            for	(int i = 2000; i<2100; i++)//    for (int i =0; i<10; i++)
//                cout<< respons[i] <<" ";
    }

    difference = clock() - before;
    msec = difference * 1000 / CLOCKS_PER_SEC;
    cout <<" calc respons raw time = " << msec <<endl;
             for (int i = 2000; i<2100; i++)//    for (int i =0; i<10; i++)
                cout<< respons[i] <<" ";


    //thrust::replace(thrust::device, respons.begin(), respons.end(), nan, 0);
    //thrust::replace(thrust::device, respons2.begin(), respons2.end(), nan, 0);
    //cout<<"respons nan" << thrust::transform_reduce(respons.begin(), respons.end(), isnan_test(), 0, thrust::plus<bool>())<< endl;
    //cout<<"respons2 nan" << thrust::transform_reduce(respons2.begin(), respons2.end(), isnan_test(), 0, thrust::plus<bool>())<< endl;


    //thrust::transform(respons2.begin(), respons2.end(),
    //                respons.begin(),respons.begin(),thrust::minus<float>());

    //cout<<"diff = " << thrust::transform_reduce(respons.begin(), respons.end(), isnan_test(), 0, thrust::plus<bool>()) << endl;

    //cuda_copy_real_to_host(respons,d_respons,N_slices*N_images);
    //sprintf(filename_buffer, "%s/responsabilities_rawkernel_%.4d.h5", conf.output_dir, count);
    // write_2d_real_array_hdf5_transpose(filename_buffer, respons, N_slices, N_images);

}

