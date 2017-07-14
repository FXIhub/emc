#include "emc.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>
#include <cufft.h>
#include <math.h>

__global__ void update_slices_kernel(real * images, real * slices, int * mask, real * respons,
				     real * scaling, int * active_images, int N_images,
				     int slice_start, int N_2d,
				     real * slices_total_respons, real * rot,
				     real * x_coord, real * y_coord, real * z_coord,
				     real * model, int slice_rows, int slice_cols,
				     int model_x, int model_y, int model_z);

__global__ void update_slices_final_kernel(real * images, real * slices, int * mask, real * respons,
					   real * scaling, int * active_images, int N_images,
					   int slice_start, int N_2d,
					   real * slices_total_respons, real * rot,
					   real * x_coord, real * y_coord, real * z_coord,
					   real * model, real * weight,
					   int slice_rows, int slice_cols,
					   int model_x, int model_y, int model_z);

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
					   real * scaling, int N_images, int N_2d,
					   real * slices_total_respons, real * rot,
					   real * x_coord, real * y_coord, real * z_coord,
					   real * model, real * weight,
					   int slice_rows, int slice_cols,
					   int model_x, int model_y, int model_z);

__global__ void insert_slices_final_kernel(real * images, real * slices, int * mask, real * respons,
					   real * scaling, int N_images, int N_2d,
					   real * slices_total_respons, real * rot,
					   real * x_coord, real * y_coord, real * z_coord,
					   real * model, real * weight,
					   int slice_rows, int slice_cols,
					   int model_x, int model_y, int model_z);

__global__ void calculate_fit_kernel(real *slices, real *images, int *mask,
				     real *respons, real *fit, real sigma,
				     real *scaling, int N_2d, int slice_start);

__global__ void calculate_fit_best_rot_kernel(real *slices, real *images, int *mask,
					      int *best_rot, real *fit,
					      real *scaling, int N_2d, int slice_start);

__global__ void calculate_radial_fit_kernel(real *slices, real *images, int *mask,
					    real *respons, real *scaling, real *radial_fit,
					    real *radial_fit_weight, real *radius,
					    int N_2d, int side, int slice_start);

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

template<typename T>
__device__ void inblock_maximum(T * data){
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

template<typename T>
__device__ void inblock_maximum_index(T * data, int *index) {
  __syncthreads();
  for (unsigned int s=blockDim.x/2; s>0; s>>=1){
    if (threadIdx.x < s){
      if (data[threadIdx.x] < data[threadIdx.x + s]) {
	data[threadIdx.x] = data[threadIdx.x + s];
	index[threadIdx.x] = index[threadIdx.x + s];
      }
    }
    __syncthreads();
  }
}

__device__ void cuda_get_slice(real *model, real *slice,
			       real *rot, real *x_coordinates,
			       real *y_coordinates, real *z_coordinates, int slice_rows,
			       int slice_cols, int model_x, int model_y, int model_z,
			       int tid, int step)
{
  const int x_max = slice_rows;
  const int y_max = slice_cols;
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
  real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
  real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
  real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
  real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
  real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
  real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
  real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
  real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
  for (int x = 0; x < x_max; x++) {
    for (int y = tid; y < y_max; y+=step) {
      /* This is just a matrix multiplication with rot */
      new_x = m00*x_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*z_coordinates[y*x_max+x];
      new_y = m10*x_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*z_coordinates[y*x_max+x];
      new_z = m20*x_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*z_coordinates[y*x_max+x];
      /* changed the next lines +0.5 -> -0.5 (11 dec 2012)*/
      round_x = lroundf(model_x/2.0f - 0.5f + new_x);
      round_y = lroundf(model_y/2.0f - 0.5f + new_y);
      round_z = lroundf(model_z/2.0f - 0.5f + new_z);
      if (round_x > 0 && round_x < model_x &&
	  round_y > 0 && round_y < model_y &&
	  round_z > 0 && round_z < model_z) {
	slice[y*x_max+x] = model[round_z*model_x*model_y + round_y*model_x + round_x];
      }else{
	slice[y*x_max+x] = -1.0f;
      }
    }
  }
}

__device__ real interpolate_model_get(real *model, int model_x, int model_y, int model_z, real new_x, real new_y, real new_z) {
  real interp_sum, interp_weight;
  real weight_x, weight_y, weight_z;
  int index_x, index_y, index_z;
  real low_weight_x, low_weight_y, low_weight_z;
  int low_x, low_y, low_z;
  int out_of_range = 0;

  if (new_x > -0.5 && new_x <= 0.) {
    low_weight_x = 0.;
    low_x = -1;
  } else if (new_x > 0. && new_x <= (model_x-1)) {
    low_weight_x = ceil(new_x) - new_x;
    low_x = (int)ceil(new_x) - 1;
  } else if (new_x > (model_x-1) && new_x < (model_x-0.5)) {
    low_weight_x = 1.;
    low_x = model_x-1;
  } else {
    out_of_range = 1;
  }

  if (new_y > -0.5 && new_y <= 0.) {
    low_weight_y = 0.;
    low_y = -1;
  } else if (new_y > 0. && new_y <= (model_y-1)) {
    low_weight_y = ceil(new_y) - new_y;
    low_y = (int)ceil(new_y) - 1;
  } else if (new_y > (model_y-1) && new_y < (model_y-0.5)) {
    low_weight_y = 1.;
    low_y = model_y-1;
  } else {
    out_of_range = 1;
  }

  if (new_z > -0.5 && new_z <= 0.) {
    low_weight_z = 0.;
    low_z = -1;
  } else if (new_z > 0. && new_z <= (model_z-1)) {
    low_weight_z = ceil(new_z) - new_z;
    low_z = (int)ceil(new_z) - 1;
  } else if (new_z > (model_z-1) && new_z < (model_z-0.5)) {
    low_weight_z = 1.;
    low_z = model_z-1;
  } else {
    out_of_range = 1;
  }

  if (out_of_range == 0) {

    interp_sum = 0.;
    interp_weight = 0.;
    for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
      if (index_x == low_x && low_weight_x == 0.) continue;
      if (index_x == (low_x+1) && low_weight_x == 1.) continue;
      if (index_x == low_x) weight_x = low_weight_x;
      else weight_x = 1. - low_weight_x;

      for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
	if (index_y == low_y && low_weight_y == 0.) continue;
	if (index_y == (low_y+1) && low_weight_y == 1.) continue;
	if (index_y == low_y) weight_y = low_weight_y;
	else weight_y = 1. - low_weight_y;

	for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
	  if (index_z == low_z && low_weight_z == 0.) continue;
	  if (index_z == (low_z+1) && low_weight_z == 1.) continue;
	  if (index_z == low_z) weight_z = low_weight_z;
	  else weight_z = 1. - low_weight_z;
	  
	  if (model[model_x*model_y*index_z + model_x*index_y + index_x] >= 0.) {
	    interp_sum += weight_x*weight_y*weight_z*model[model_x*model_y*index_z + model_x*index_y + index_x];
	    interp_weight += weight_x*weight_y*weight_z;
	  }
	}
      }
    }
    if (interp_weight > 0.) {
      return interp_sum / interp_weight;
    } else {
      return -1.0f;
    }
  } else {
    return -1.0f;
  }
}

__device__ void cuda_get_slice_interpolate(real *model, real *slice, real *rot,
					   real *x_coordinates, real *y_coordinates, real *z_coordinates,
					   int slice_rows, int slice_cols, int model_x, int model_y, int model_z,
					   int tid, int step) {
  const int x_max = slice_rows;
  const int y_max = slice_cols;
  //tabulate angle later
  real new_x, new_y, new_z;

  real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
  real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
  real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
  real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
  real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
  real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
  real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
  real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
  real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
  for (int x = 0; x < x_max; x++) {
    for (int y = tid; y < y_max; y+=step) {
      /* This is just a matrix multiplication with rot */
      new_x = m00*x_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*z_coordinates[y*x_max+x] + model_x/2.0 - 0.5;
      new_y = m10*x_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*z_coordinates[y*x_max+x] + model_y/2.0 - 0.5;
      new_z = m20*x_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*z_coordinates[y*x_max+x] + model_z/2.0 - 0.5;

      slice[y*x_max+x] = interpolate_model_get(model, model_x, model_y, model_z, new_x, new_y, new_z);
    }
  }
}

/* updated to use rotations with an offset start. */
__global__ void get_slices_kernel(real * model, real * slices, real *rot, real *x_coordinates,
				  real *y_coordinates, real *z_coordinates, int slice_rows,
				  int slice_cols, int model_x, int model_y, int model_z,
				  int start_slice){
  int bid = blockIdx.x;
  int i_slice = bid;
  int tid = threadIdx.x;
  int step = blockDim.x;
  int N_2d = slice_rows*slice_cols;
  /*
  cuda_get_slice(model,&slices[N_2d*i_slice],&rot[4*(start_slice+i_slice)],x_coordinates,
		 y_coordinates,z_coordinates,slice_rows,slice_cols,model_x,model_y,
		 model_z,tid,step);
  */

  cuda_get_slice_interpolate(model,&slices[N_2d*i_slice],&rot[4*(start_slice+i_slice)],x_coordinates,
			     y_coordinates,z_coordinates,slice_rows,slice_cols,model_x,model_y,
			     model_z,tid,step);


}

__global__ void cuda_test_interpolate_kernel(real *model, int side, real *return_value) {
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
  //model[side*side*4 + side*4 + 5] = -1.;
  //model[side*side*4 + side*5 + 5] = -1.;

  real interp_x = 4.5;
  real interp_y = 4.5;
  real interp_z = 4.5;
  return_value[0] = interpolate_model_get(model, side, side, side, interp_x, interp_y, interp_z);
  printf("interpolate at %g %g %g -> %g\n", interp_x, interp_y, interp_z, interpolate_model_get(model, side, side, side, interp_x, interp_y, interp_z));
}

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

/* This responsability does not yet take scaling of patterns into accoutnt. */
__device__ void cuda_calculate_responsability_absolute(float *slice, float *image, int *mask, real *weight_map,
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

__device__ void cuda_calculate_responsability_relative(float *slice, float *image, int *mask, real *weight_map,
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
__device__ void cuda_calculate_responsability_poisson(float *slice, float *image, int *mask, real *weight_map,
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
  count_cache[tid] = count;
  //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_responsability_true_poisson(float *slice, float *image,
							   int *mask, real sigma, real scaling, real *weight_map,
							   int N_2d, int tid, int step,
							   real * sum_cache, real * count_cache)
{
  real sum = 0.0;
  const int i_max = N_2d;
  real count = 0;
  for (int i = tid; i < i_max; i+=step) {
    if (mask[i] != 0 && slice[i] > 0.0f) {
      //sum += (pow((slice[i]*scaling - image[i]) / 8.0, 2) / (image[i]/8.0 + 0.1) / 2.0) * weight_map[i];
      sum += logf(sqrt(slice[i])) + pow((slice[i] - image[i]/scaling), 2)/slice[i]*weight_map[i];
      //sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+1.0), 2);
      count += weight_map[i];
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count;
  //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}
/*
__device__ void cuda_calculate_responsability_annealing_poisson(float *slice, float *image, int *mask, real sigma,
								real scaling, real *weight_map, int N_2d, int tid,
								int step, real *sum_cache, real *count_cache) {
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
*/

/*
__device__ void cuda_calculate_responsability_adaptive(float *slice, float *image,
						       int *mask, real sigma, real scaling, real *weight_map,
						       int N_2d, int tid, int step,
						       real *sum_cache, real *count_cache)
{
  real sum = 0.0;
  const int i_max = N_2d;
  real count = 0;
  for (int i = tid; i < i_max; i+=step) {
    if (mask[i] != 0 && slice[i] > 0.0f) {
      sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]), 2)*weight_map[i];
      count += weight_map[i];
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count
}
*/

/* Now takes a starting slice. Otherwise unchanged */
__global__ void calculate_responsabilities_kernel(float * slices, float * images, int * masks, real *weight_map,
						  real sigma, real * scaling, real * respons, real *weights, 
						  int N_2d, int slice_start, enum diff_type diff){
  __shared__ real sum_cache[256];
  __shared__ real count_cache[256];
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;

  if (diff == relative) {
    cuda_calculate_responsability_relative(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[N_2d*i_image], weight_map,
					   sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
					   sum_cache, count_cache);
  } else if (diff == poisson) {
    cuda_calculate_responsability_poisson(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[N_2d*i_image], weight_map,
					  sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
					  sum_cache, count_cache);
  } else if (diff == absolute) {
    /* This one was used for best result so far.*/
    cuda_calculate_responsability_absolute(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[N_2d*i_image], weight_map,
					   sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step,
					   sum_cache, count_cache);
  } else if (diff == annealing_poisson) {
    printf("annealing_poisson is not implemented\n");
    return;
    /*
    cuda_calculate_responsability_annealing_poisson(&slices[i_slice*N_2d], &images[i_image*N_2d], mask, weight_map,
						    sigma,scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid,
						    step, sum_cache, count_cache);
    */
  }

  inblock_reduce(sum_cache);
  inblock_reduce(count_cache);
  
  __syncthreads(); //probably not needed
  if(tid == 0){
    //respons[(slice_start+i_slice)*N_images+i_image] = -sum_cache[0]/2.0/(real)count_cache[0]/pow(sigma,2);
    /* This weight buiseniss is fishy. It seems wee are getting responsabilities that are lower close to vertices. */
    //respons[(slice_start+i_slice)*N_images+i_image] = log(weights[slice_start+i_slice]) - sum_cache[0]/2.0/count_cache[0]/pow(sigma,2);
    /* I therefore try to remove it altogether */
    if (count_cache[0] > 0) {
      respons[(slice_start+i_slice)*N_images+i_image] = -sum_cache[0]/2.0/count_cache[0]/pow(sigma,2);
    } else {
      respons[(slice_start+i_slice)*N_images+i_image] = 10.;
    }
  }
}


/* Now takes start slice and slice chunk. Also removed memcopy, done separetely later. */
void cuda_calculate_responsabilities(real * d_slices, real * d_images, int * d_masks, real *d_weight_map,
				     real sigma, real * d_scaling, real * d_respons, real *d_weights, 
				     int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff){
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  dim3 nblocks(N_images,slice_chunk);
  int nthreads = 256;
  calculate_responsabilities_kernel<<<nblocks,nthreads>>>(d_slices, d_images, d_masks, d_weight_map,
							  sigma, d_scaling, d_respons, d_weights,
							  N_2d, slice_start, diff);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (calc resp): %s\n",cudaGetErrorString(status));
  }

  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda calculate_responsabilities time = %fms\n",k_ms);
}
  
void cuda_calculate_responsabilities_sum(real * respons, real * d_respons, int N_slices,
					 int N_images){
  cudaMemcpy(respons,d_respons,sizeof(real)*N_slices*N_images,cudaMemcpyDeviceToHost);
  real respons_sum = 0;
  for(int i = 0;i<N_slices*N_images;i++){
    if (respons[i] < 0.) { // Positive values signals invalid responsability
      respons_sum += respons[i];
    }
  }
  printf("respons_sum = %f\n",respons_sum);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (resp sum): %s\n",cudaGetErrorString(status));
  }
}  

__global__ void calculate_weighted_power_kernel(real * images, real * slices, int * mask,
						real *respons, real * weighted_power, int N_images,
						int slice_start, int slice_chunk, int N_2d) {
  __shared__ real correlation[256];
  //__shared__ int count[256];
  int step = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int i_image = bid;
  for (int i_slice = 0; i_slice < slice_chunk; i_slice++) { 
    correlation[tid] = 0.0;
    //count[tid] = 0;
    for (int i = tid; i < N_2d; i+=step) {
      if (mask[i] != 0 && slices[i_slice*N_2d+i] > 0.0f) {
	correlation[tid] += images[i_image*N_2d+i]*slices[i_slice*N_2d+i];
	//correlation[tid] += images[i_image*N_2d+i]/slices[i_slice*N_2d+i];
	//count[tid] += 1;
      }
    }
    inblock_reduce(correlation);
    //inblock_reduce(count);
    if(tid == 0){
      weighted_power[i_image] += respons[(slice_start+i_slice)*N_images+i_image]*correlation[tid];
      //weighted_power[i_image] += correlation[tid]/count[tid]*respons[(slice_start+i_slice)*N_images+i_image];
    }
  }
}

__global__ void slice_weighting_kernel(real * images,int * masks,
				       real * scaling, real *weighted_power,
				       int N_slices, int N_2d){
  __shared__ real image_power[256];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = bid;  
  // make sure weighted power is set to 0


  image_power[tid] = 0.0;
  for (int i = tid; i < N_2d; i+=step) {
    if (masks[i_image*N_2d+i] != 0) {
      image_power[tid] += pow(images[i_image*N_2d+i],2);
    }
  }
  inblock_reduce(image_power);

  if(tid == 0){
    scaling[i_image] = image_power[tid]/weighted_power[i_image];
    //scaling[i_image] = weighted_power[i_image];
  }
}

void cuda_update_weighted_power(real * d_images, real * d_slices, int * d_mask,
				real * d_respons, real * d_weighted_power, int N_images,
				int slice_start, int slice_chunk, int N_2d) {
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  int nblocks = N_images;
  int nthreads = 256;
  calculate_weighted_power_kernel<<<nblocks,nthreads>>>(d_images,d_slices,d_mask,
							d_respons,d_weighted_power, N_images,
							slice_start,slice_chunk,N_2d);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error: %s\n",cudaGetErrorString(status));
  }

  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda calculate weighted power time = %fms\n",k_ms);
}

void cuda_update_scaling(real * d_images, int * d_masks,
			 real * d_scaling, real *d_weighted_power, int N_images,
			 int N_slices, int N_2d, real * scaling){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  int nblocks = N_images;
  int nthreads = 256;
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);
  slice_weighting_kernel<<<nblocks,nthreads>>>(d_images,d_masks,d_scaling,
					       d_weighted_power,N_slices,N_2d);
  cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);
  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda kernel update scaling time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (update scaling): %s\n",cudaGetErrorString(status));
  }
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  real ms;
  cudaEventElapsedTime (&ms, begin, end);
  //printf("cuda update scaling time = %fms\n",ms);
}

__global__ void calculate_best_rotation_kernel(real *respons, int *best_rotation, int N_slices) {
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int N_images = gridDim.x;
  
  __shared__ real max_resp[256];
  __shared__ int max_index[256];
  max_resp[tid] = -1.e100;
  max_index[tid] = 0;
  real this_resp;
  for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
    this_resp = respons[i_slice*N_images+i_image];
    if (this_resp > max_resp[tid]) {
      //printf("new best resp found at %d\n", i_slice);
      max_resp[tid] = this_resp;
      max_index[tid] = i_slice;
      //printf("max_index set to %d\n", max_index[tid]);
    }
  }
  //printf("before reduce max_index[%d] = %d %g)\n", tid, max_index[tid], max_resp[tid]);
  inblock_maximum_index(max_resp, max_index);
  if (tid == 0) {
    best_rotation[i_image] = max_index[0];
    //if (i_image < 4) {
    //printf("best_rotation[%d] = %d (%g)\n", i_image, best_rotation[i_image], respons[best_rotation[i_image]*N_images+i_image]);
      //}
  }
}

__device__ real calculate_scaling_poisson(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step){
  __shared__ real sum_cache[256];
  __shared__ real weight_cache[256];
  sum_cache[tid] = 0.;
  weight_cache[tid] = 0; 
  for (int i = tid; i < N_2d; i+=step) {
    if (mask[i] > 0 && slice[i] > 1.e-10) {
      //if (mask[i] > 0) {
      /*
      sum_cache[tid] += image[i] / slice[i];
      weight_cache[tid] += 1.;
      */
      sum_cache[tid] += image[i]*image[i]/slice[i]*weight_map[i];
      weight_cache[tid] += image[i]*weight_map[i];
    }
  }
  inblock_reduce(sum_cache);
  inblock_reduce(weight_cache);
  __syncthreads();
  return sum_cache[0] / weight_cache[0];
}

__device__ real calculate_scaling_absolute(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step){
  __shared__ real sum_cache[256];
  __shared__ real weight_cache[256];
  sum_cache[tid] = 0.;
  weight_cache[tid] = 0; 
  for (int i = tid; i < N_2d; i+=step) {
    if (mask[i] > 0 && slice[i] > 1.e-10 && image[i] > 1.e-10) {
      //if (mask[i] > 0) {
      /*
      sum_cache[tid] += image[i] / slice[i];
      weight_cache[tid] += 1.;
      */
      sum_cache[tid] += image[i]*image[i]*weight_map[i];
      weight_cache[tid] += image[i]*slice[i]*weight_map[i];
    }
  }
  inblock_reduce(sum_cache);
  inblock_reduce(weight_cache);
  __syncthreads();
  return sum_cache[0] / weight_cache[0];
}

__device__ real calculate_scaling_relative(real *image, real *slice, int *mask, real *weight_map, int N_2d, int tid, int step){
  __shared__ real sum_cache[256];
  __shared__ real weight_cache[256];
  sum_cache[tid] = 0.;
  weight_cache[tid] = 0; 
  for (int i = tid; i < N_2d; i+=step) {
    if (mask[i] > 0 && slice[i] > 1.e-10) {
      //if (mask[i] > 0) {
      /*
      sum_cache[tid] += image[i] / slice[i];
      weight_cache[tid] += 1.;
      */
      sum_cache[tid] += image[i]*image[i]/(slice[i]*slice[i]) * weight_map[i];
      //weight_cache[tid] += image[i]/slice[i];
      weight_cache[tid] += image[i]/slice[i] * weight_map[i];
    }
  }
  inblock_reduce(sum_cache);
  inblock_reduce(weight_cache);
  __syncthreads();
  return sum_cache[0] / weight_cache[0];
}

__global__ void update_scaling_best_kernel(real *scaling, real *images, real *model, int *mask, real *weight_map, real *rotations,
					   real *x_coordinates, real *y_coordinates, real *z_coordinates,
					   int side, int *best_rotation){
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int tid = threadIdx.x;
  const int N_2d = side*side;
  extern __shared__ real this_slice[];
  /*
  if (tid == 0) {
    printf("best_rotation[%d] = %d\n", i_image, best_rotation[i_image]);
  }
  */
  
  cuda_get_slice(model, this_slice, &rotations[4*best_rotation[i_image]],
		 x_coordinates, y_coordinates, z_coordinates,
		 side, side, side, side, side, tid, step);
  /*
  if (tid == 0) {
    printf("slice value [%d] = %g\n", i_image, this_slice[20*64 + 20]);
  }
  */

  real this_scaling = calculate_scaling_poisson(&images[N_2d*i_image], this_slice, mask, weight_map, N_2d, tid, step);
  if (tid == 0) {
    scaling[i_image] = this_scaling;
  }
}

void cuda_update_scaling_best(real *d_images, int *d_mask,
			      real *d_model, real *d_scaling, real *d_weight_map, real *d_respons, real *d_rotations,
			      real *x_coordinates, real *y_coordinates, real *z_coordinates,
			      int N_images, int N_slices, int side, real *scaling) {
  int nblocks = N_images;
  int nthreads = 256;
  const int N_2d = side*side;
  int *d_best_rotation;
  cudaMalloc(&d_best_rotation, N_images*sizeof(int));
  calculate_best_rotation_kernel<<<nblocks, nthreads>>>(d_respons, d_best_rotation, N_slices);
  nthreads = 256;
  nblocks = N_images;
  update_scaling_best_kernel<<<nblocks,nthreads,N_2d*sizeof(real)>>>(d_scaling, d_images, d_model, d_mask, d_weight_map, d_rotations, x_coordinates, y_coordinates, z_coordinates, side, d_best_rotation);
  cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);
}

__global__ void update_scaling_full_kernel(real *images, real *slices, int *masks, real *scaling, real *weight_map, int N_2d, int slice_start, enum diff_type diff) {
  const int tid = threadIdx.x;
  const int step = blockDim.x;
  const int i_image = blockIdx.x;
  const int i_slice = blockIdx.y;
  const int N_images = gridDim.x;
  real this_scaling;
  /*
  this_scaling = calculate_scaling_poisson(&images[N_2d*i_image], &slices[N_2d*i_slice], mask, weight_map, N_2d, tid, step);
  */

  if (diff == poisson) {
    this_scaling = calculate_scaling_poisson(&images[N_2d*i_image], &slices[N_2d*i_slice], &masks[N_2d*i_image], weight_map, N_2d, tid, step);
  } else if (diff == absolute) {
    this_scaling = calculate_scaling_absolute(&images[N_2d*i_image], &slices[N_2d*i_slice], &masks[N_2d*i_image], weight_map, N_2d, tid, step);
  } else if (diff == relative) {
    this_scaling = calculate_scaling_relative(&images[N_2d*i_image], &slices[N_2d*i_slice], &masks[N_2d*i_image], weight_map, N_2d, tid, step);
  }

  //this_scaling *= 0.5;
  __syncthreads();
  if (tid == 0) {
    scaling[(slice_start+i_slice)*N_images+i_image] = this_scaling;
  }
}

void cuda_update_scaling_full(real *d_images, real *d_slices, int *d_masks, real *d_scaling, real *d_weight_map,
			      int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff) {
  dim3 nblocks(N_images,slice_chunk);
  int nthreads = 256;
  update_scaling_full_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_masks, d_scaling, d_weight_map, N_2d, slice_start, diff);
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

void cuda_update_slices(real * d_images, real * d_slices, int * d_masks,
			real * d_respons, real * d_scaling, int * d_active_images, int N_images,
			int slice_start, int slice_chunk, int N_2d,
			sp_3matrix * model, real * d_model,
			real *d_x_coordinates, real *d_y_coordinates,
			real *d_z_coordinates, real *d_rot,
			real * d_weight, sp_matrix ** images){
  dim3 nblocks = slice_chunk;//N_slices;
  int nthreads = 256;
  real * d_slices_total_respons;
  cudaMalloc(&d_slices_total_respons,sizeof(real)*slice_chunk);

  /*
  real * d_weights;
  cudaMalloc(&d_weights,sizeof(real)*slice_chunk);
  cudaMemcpy(d_weights,weights,sizeof(real)*slice_chunk,cudaMemcpyHostToDevice);
  */

  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  update_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_masks, d_respons,
					     d_scaling, d_active_images, N_images, slice_start, N_2d,
					     d_slices_total_respons, d_rot,d_x_coordinates,
					     d_y_coordinates,d_z_coordinates,d_model,
					     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
					     sp_3matrix_x(model),sp_3matrix_y(model),
					     sp_3matrix_z(model));  
  cudaThreadSynchronize();
  insert_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_masks, d_respons,
					     d_scaling, N_images, N_2d,
					     d_slices_total_respons, d_rot,d_x_coordinates,
					     d_y_coordinates,d_z_coordinates,d_model, d_weight,
					     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
					     sp_3matrix_x(model),sp_3matrix_y(model),
					     sp_3matrix_z(model));

  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda kernel slice update time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (update slices): %s\n",cudaGetErrorString(status));
  }
  cudaFree(d_slices_total_respons);
}

void cuda_update_slices_final(real * d_images, real * d_slices, int * d_masks,
			real * d_respons, real * d_scaling, int * d_active_images, int N_images,
			int slice_start, int slice_chunk, int N_2d,
			sp_3matrix * model, real * d_model,
			real *d_x_coordinates, real *d_y_coordinates,
			real *d_z_coordinates, real *d_rot,
			real * d_weight, sp_matrix ** images){
  dim3 nblocks = slice_chunk;//N_slices;
  int nthreads = 256;
  real * d_slices_total_respons;
  cudaMalloc(&d_slices_total_respons,sizeof(real)*slice_chunk);
  /*
  real * d_weights;
  cudaMalloc(&d_weights,sizeof(real)*slice_chunk);
  cudaMemcpy(d_weights,weights,sizeof(real)*slice_chunk,cudaMemcpyHostToDevice);
  */

  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  update_slices_final_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_masks, d_respons,
						   d_scaling, d_active_images, N_images, slice_start, N_2d,
						   d_slices_total_respons, d_rot,d_x_coordinates,
						   d_y_coordinates,d_z_coordinates,d_model, d_weight,
						   sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
						   sp_3matrix_x(model),sp_3matrix_y(model),
						   sp_3matrix_z(model));

  cudaThreadSynchronize();
  //cudaMemcpy(h_slices,d_slices,N_2d*sizeof(real)*slice_chunk,cudaMemcpyDeviceToHost);
  insert_slices_final_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_masks, d_respons,
					     d_scaling, N_images, N_2d,
					     d_slices_total_respons, d_rot,d_x_coordinates,
					     d_y_coordinates,d_z_coordinates,d_model, d_weight,
					     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
					     sp_3matrix_x(model),sp_3matrix_y(model),
					     sp_3matrix_z(model));
  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda kernel slice update time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (update slices): %s\n",cudaGetErrorString(status));
  }
}

real cuda_model_max(real * model, int model_size){
  thrust::device_ptr<real> p(model);
  real max = thrust::reduce(p, p+model_size, real(0), thrust::maximum<real>());
  return max;
}

__global__ void model_average_kernel(real *model, int model_size, real *average) {
  const int tid = threadIdx.x;
  const int step = blockDim.x;
  //const int i1 = blockIdx.x;
  __shared__ real sum_cache[256];
  __shared__ int weight_cache[256];
  sum_cache[tid] = 0.;
  weight_cache[tid] = 0;
  for (int i = tid; i < model_size; i+=step) {
    if (model[i] >= 0.) {
      sum_cache[tid] += model[i];
      weight_cache[tid] += 1;
    }
  }
  inblock_reduce(sum_cache);
  inblock_reduce(weight_cache);
  __syncthreads();
  if (tid == 0) {
    *average = sum_cache[0] / weight_cache[0];
  }
}

real cuda_model_average(real * model, int model_size) {
  /*
  thrust::device_ptr<real> p(model);
  real sum = thrust::reduce(p, p+model_size, real(0), thrust::plus<real>());
  return sum;
  */
  real *d_average;
  cudaMalloc(&d_average, sizeof(real));
  model_average_kernel<<<1,256>>>(model, model_size, d_average);
  real average;
  cudaMemcpy(&average, d_average, sizeof(real), cudaMemcpyDeviceToHost);
  cudaFree(d_average);
  return average;
}

void cuda_allocate_slices(real ** slices, int side, int N_slices){
  //cudaSetDevice(2);
  cudaMalloc(slices,sizeof(real)*side*side*N_slices);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_print_device_info): %s\n",cudaGetErrorString(status));
  }
}

void cuda_allocate_model(real ** d_model, sp_3matrix * model){
  cudaMalloc(d_model,sizeof(real)*sp_3matrix_size(model));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_model: malloc): %s\n",cudaGetErrorString(status));
  }
  cudaMemcpy(*d_model,model->data,sizeof(real)*sp_3matrix_size(model),cudaMemcpyHostToDevice);
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_model: copy): %s\n",cudaGetErrorString(status));
  }
}

void cuda_allocate_mask(int ** d_mask, sp_imatrix * mask){
  cudaMalloc(d_mask,sizeof(int)*sp_imatrix_size(mask));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_mask: malloc): %s\n",cudaGetErrorString(status));
  }

  cudaMemcpy(*d_mask,mask->data,sizeof(int)*sp_imatrix_size(mask),cudaMemcpyHostToDevice);
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_mask: copy): %s\n",cudaGetErrorString(status));
  }
}

void cuda_allocate_rotations(real ** d_rotations, Quaternion *rotations,  int N_slices){
  cudaMalloc(d_rotations, sizeof(real)*4*N_slices);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_rotations: malloc): %s\n",cudaGetErrorString(status));
  }

  /*
  for(int i = 0; i<N_slices; i++){
    cudaMemcpy(&((*d_rotations)[4*i]),rotations[i]->q,sizeof(real)*4,cudaMemcpyHostToDevice);
  }
  */
  cudaMemcpy(*d_rotations, rotations, sizeof(real)*4*N_slices, cudaMemcpyHostToDevice);
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_rotations: copy): %s\n",cudaGetErrorString(status));
  }
}

void cuda_allocate_images(real ** d_images, sp_matrix ** images,  int N_images){

  cudaMalloc(d_images,sizeof(real)*sp_matrix_size(images[0])*N_images);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_images: malloc): %s\n",cudaGetErrorString(status));
  }

  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(*d_images)[sp_matrix_size(images[0])*i],images[i]->data,sizeof(real)*sp_matrix_size(images[0]),cudaMemcpyHostToDevice);
  }
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_images: copy): %s\n",cudaGetErrorString(status));
  }

}

void cuda_allocate_individual_masks(int ** d_masks, sp_imatrix ** masks, int N_images){
  cudaMalloc(d_masks,sizeof(int)*sp_imatrix_size(masks[0])*N_images);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_individual_masks: malloc): %s\n",cudaGetErrorString(status));
  }
  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(*d_masks)[sp_imatrix_size(masks[0])*i],masks[i]->data,sizeof(int)*sp_imatrix_size(masks[0]),cudaMemcpyHostToDevice);
  }
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_individual_masks: copy): %s\n",cudaGetErrorString(status));
  }
}


void cuda_allocate_common_masks(int ** d_masks, sp_imatrix *mask, int N_images){
  cudaMalloc(d_masks,sizeof(int)*sp_imatrix_size(mask)*N_images);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_common_masks: malloc): %s\n",cudaGetErrorString(status));
  }
  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(*d_masks)[sp_imatrix_size(mask)*i],mask->data,sizeof(int)*sp_imatrix_size(mask),cudaMemcpyHostToDevice);
  }
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_common_masks: copy): %s\n",cudaGetErrorString(status));
  }
}


__global__ void apply_mask(real *const array, const int *const mask, const int size) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < size) {
    if (mask[i] == 0) {
      array[i] = -1.;
    }
  }
}

/* This function is the same as apply_mask except that it
   periodically steps throug the mask thus applying the same
   mask to multiple files. */
__global__ void apply_single_mask(real * const array, const int *const mask, const int mask_size, const int size) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < size) {
    if (mask[i%mask_size] == 0) {
      array[i] = -1.;
    }
  }
}

void cuda_apply_masks(real *const d_images, const int *const d_masks, const int N_2d, const int N_images) {
  int nthreads = 256;
  int nblocks = (N_2d*N_images - 1) / nthreads;
  apply_mask<<<nblocks, nthreads>>>(d_images, d_masks, N_2d*N_images);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_apply_masks): %s\n",cudaGetErrorString(status));
  }
}

void cuda_apply_single_mask(real *const d_images, const int *const d_mask, const int N_2d, const int N_images) {
  int nthreads = 256;
  int nblocks = (N_2d*N_images - 1) / nthreads;
  apply_single_mask<<<nblocks, nthreads>>>(d_images, d_mask, N_2d, N_2d*N_images);
}

void cuda_allocate_coords(real ** d_x, real ** d_y, real ** d_z, sp_matrix * x,
			  sp_matrix * y, sp_matrix * z){
  cudaMalloc(d_x,sizeof(real)*sp_matrix_size(x));
  cudaMalloc(d_y,sizeof(real)*sp_matrix_size(x));
  cudaMalloc(d_z,sizeof(real)*sp_matrix_size(x));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_coords: malloc): %s\n",cudaGetErrorString(status));
  }

  cudaMemcpy(*d_x,x->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
  cudaMemcpy(*d_y,y->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
  cudaMemcpy(*d_z,z->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
  status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_coords: copy): %s\n",cudaGetErrorString(status));
  }
}

void cuda_reset_model(sp_3matrix * model, real * d_model){
  cudaMemset(d_model,0,sizeof(real)*sp_3matrix_size(model));
}

void cuda_copy_model(sp_3matrix * model, real *d_model){
  cudaMemcpy(model->data,d_model,sizeof(real)*sp_3matrix_size(model),cudaMemcpyDeviceToHost);
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


__global__ void cuda_divide_model_kernel(real * model, real * weight, int n){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < n) {
    if(weight[i] > 0.0f){
      model[i] /= weight[i];
    }else{
      //model[i] = 0.0f;
      model[i] = -1.f;
    }
  }
}

__global__ void cuda_mask_out_model_kernel(real *model, real *weight, int n){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < n) {
    if(weight[i] <= 0.0f){
      model[i] = -1.0f;
    }
  }
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
  printf("Using new normalization!\n");
  cudaMemcpy(model->data, d_model, sp_3matrix_size(model)*sizeof(real), cudaMemcpyDeviceToHost);
  const int model_size = sp_3matrix_size(model);
  real model_average = 0.;
  real model_count = 0.;
  for (int index = 0; index < model_size; index++) {
    if (model->data[index] >= 0.) {
      model_average += model->data[index];
      model_count += 1.;
    }
  }
  if (model_count > 0.) {
    model_average /= model_count;
  } else {
    model_average = 1.;
  }
  
  thrust::device_ptr<real> p(d_model);
  //real model_average = cuda_model_average(d_model, sp_3matrix_size(model));
    
  printf("model average before normalization = %g\n", model_average);
  //real model_sum = thrust::reduce(p, p+n, real(0), thrust::plus<real>());
  //model_sum /= (real) n;
  thrust::transform(p, p+model_size,thrust::make_constant_iterator(1.0f/model_average), p, thrust::multiplies<real>());
}

void cuda_print_device_info() {
  int i_device = cuda_get_device();
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, i_device);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_print_device_info): %s\n",cudaGetErrorString(status));
  }
  
  printf("Name: %s\n", properties.name);
  printf("Compute Capability: %d.%d\n", properties.major, properties.minor);
  printf("Memory: %g GB\n", properties.totalGlobalMem/(1024.*1024.*1024.));
  printf("Number of cores: %d\n", 8*properties.multiProcessorCount);

}

int cuda_get_best_device() {
  int N_devices;
  cudaDeviceProp properties;
  cudaGetDeviceCount(&N_devices);
  int core_count = 0;
  int best_device = 0;
  for (int i_device = 0; i_device < N_devices; i_device++) {
    cudaGetDeviceProperties(&properties, i_device);
    if (properties.multiProcessorCount > core_count) {
      best_device = i_device;
      core_count = properties.multiProcessorCount;
    }
  }
  return best_device;
  //cuda_set_device(best_device);

  /* should use cudaSetValidDevices() instead */
}


int compare(const void *a, const void *b) {
  return *(int*)b - *(int*)a;
}

/* this function is much safer than cuda_get_best_device() since it works together
   with exclusive mode */
void cuda_choose_best_device() {
  int N_devices;
  cudaDeviceProp properties;
  cudaGetDeviceCount(&N_devices);
  int *core_count = (int *)malloc(N_devices*sizeof(int));
  int **core_count_pointers = (int **)malloc(N_devices*sizeof(int *));
  for (int i_device = 0; i_device < N_devices; i_device++) {
    cudaGetDeviceProperties(&properties, i_device);
    core_count[i_device] = properties.multiProcessorCount;
    core_count_pointers[i_device] = &core_count[i_device];
  }
  
  qsort(core_count_pointers, N_devices, sizeof(core_count_pointers[0]), compare);
  int *device_priority = (int *)malloc(N_devices*sizeof(int));
  for (int i_device = 0; i_device < N_devices; i_device++) {
    device_priority[i_device] = (int) (core_count_pointers[i_device] - core_count);
  }
  cudaSetValidDevices(device_priority, N_devices);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_choose_best_device): %s\n",cudaGetErrorString(status));
  }
  free(core_count_pointers);
  free(core_count);
  free(device_priority);
}

int cuda_get_device() {
  int i_device;
  cudaGetDevice(&i_device);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_get_device): %s\n",cudaGetErrorString(status));
  }
  return i_device;
}

void cuda_set_device(int i_device) {
  cudaSetDevice(i_device);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_set_device): %s\n",cudaGetErrorString(status));
  }
}

int cuda_get_number_of_devices() {
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_get_number_of_devices): %s\n",cudaGetErrorString(status));
  }
  return n_devices;
}

void cuda_allocate_real(real ** x, int n){
  cudaMalloc(x,n*sizeof(real));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_real): %s\n",cudaGetErrorString(status));
  }
}

void cuda_allocate_int(int ** x, int n){
  cudaMalloc(x,n*sizeof(real));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_int): %s\n",cudaGetErrorString(status));
  }
}

void cuda_set_to_zero(real * x, int n){
  cudaMemset(x,0.0,sizeof(real)*n);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_set_to_zero): %s\n",cudaGetErrorString(status));
  }
}

void cuda_copy_real_to_device(real *x, real *d_x, int n){
  cudaMemcpy(d_x,x,n*sizeof(real),cudaMemcpyHostToDevice);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_copy_real_to_device): %s\n",cudaGetErrorString(status));
  }
}

void cuda_copy_real_to_host(real *x, real *d_x, int n){
  cudaMemcpy(x,d_x,n*sizeof(real),cudaMemcpyDeviceToHost);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_copy_real_to_host: copy): %s\n",cudaGetErrorString(status));
  }
}

void cuda_copy_int_to_device(int *x, int *d_x, int n){
  cudaMemcpy(d_x,x,n*sizeof(int),cudaMemcpyHostToDevice);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_copy_int_to_host): %s\n",cudaGetErrorString(status));
  }
}

void cuda_copy_int_to_host(int *x, int *d_x, int n){
  cudaMemcpy(x,d_x,n*sizeof(int),cudaMemcpyDeviceToHost);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_copy_int_to_host): %s\n",cudaGetErrorString(status));
  }
}
			  
void cuda_allocate_scaling(real ** d_scaling, int N_images){
  cudaMalloc(d_scaling,N_images*sizeof(real));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_scaling): %s\n",cudaGetErrorString(status));
  }
  thrust::device_ptr<real> p(*d_scaling);
  thrust::fill(p, p+N_images, real(1));
}

void cuda_allocate_scaling_full(real **d_scaling, int N_images, int N_slices) {
  cudaMalloc(d_scaling, N_images*N_slices*sizeof(real));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_scaling_full): %s\n",cudaGetErrorString(status));
  }
  thrust::device_ptr<real> p(*d_scaling);
  thrust::fill(p, p+N_images*N_slices, real(1.));
}

__global__ void cuda_normalize_responsabilities_single_kernel(real *respons, int N_slices, int N_images) {
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

__global__ void cuda_normalize_responsabilities_uniform_kernel(real * respons, int N_slices, int N_images){
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

  /* nor normalize each images weight to one */
  /*
  int i_image = blockIdx.x;
  cache[tid] = 0;
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    if (respons[i_slice*N_images+i_image] > min_resp) {
      //respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
      cache[tid] += respons[i_slice*N_images+i_image];
    } else {
      respons[i_slice*N_images+i_image] = 0.0f;
    }
  }
  inblock_reduce(cache);
  //real sum = cache[0];
  sum = cache[0];
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    respons[i_slice*N_images+i_image] /= sum;
  }
  */
  /*
  int i_image = blockIdx.x;
  if (i_image < N_images) {
    cache[tid] = 0;
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
      cache[tid] =+ respons[i_slice*N_images+i_image];
    }
    inblock_reduce(cache);
    sum = cache[0];
    for (int i_slice = tid; i_slice < N_slices; i_slice += step) {
      respons[i_slice*N_images+i_image] /= sum;
    }
  }
  */
  const real scaling_factor = ((float) N_images) / ((float) N_slices);
  for (int i_image = tid; i_image < N_images; i_image += step) {
    respons[i_slice*N_images+i_image] *= scaling_factor;
  }
}

/* Before normalizing, take everything to the power needed to so that the normalization factor becomes x. Where x changes but typically I guess it should be about 3. This power might not be easy to calculate.*/
/*
__global__ void cuda_normalize_responsabilities_adaptive_kernel(real *respons, int N_slices, int *N_images)
{
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
  inblock_reduce(cache);
  real sum = cache[0];
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    respons[i_slice*N_images+i_image] /= sum;
  }
}
*/
__global__ void cuda_normalize_responsabilities_kernel(real * respons, int N_slices, int N_images){
  __shared__ real cache[256];

  int i_image = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  cache[tid] = -1.0e10f;
  for(int i_slice = tid;i_slice < N_slices;i_slice += step){
    if(respons[i_slice*N_images+i_image] < 0. && cache[tid] < respons[i_slice*N_images+i_image]){
      cache[tid] = respons[i_slice*N_images+i_image];
    }
  }
  inblock_maximum(cache);
  real max_resp = cache[0];
  __syncthreads();
  for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
    if (respons[i_slice*N_images+i_image] < 0.) { 
      respons[i_slice*N_images+i_image] -= max_resp;
    }
  }

  cache[tid] = 0;
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    if (respons[i_slice*N_images+i_image] < 0.) {
      if (respons[i_slice*N_images+i_image] > min_resp) {
	respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
	cache[tid] += respons[i_slice*N_images+i_image];
      } else {
	respons[i_slice*N_images+i_image] = 0.0f;
      }
    } else {
      respons[i_slice*N_images+i_image] = 0.; // Set invalid responsabilities to 0. (no weight) since -1. (invalid) is not implemented.
    }
  }
  inblock_reduce(cache);
  real sum = cache[0];
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    if (respons[i_slice*N_images+i_image] > 0.) {
      respons[i_slice*N_images+i_image] /= sum;
    }
  }
}

void cuda_normalize_responsabilities_single(real *d_respons, int N_slices, int N_images) {
  int nblocks = N_images;
  int nthreads = 256;
  
  cuda_normalize_responsabilities_single_kernel<<<nblocks, nthreads>>>(d_respons, N_slices, N_images);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    printf("CUDA Error (norm resp): %s\n", cudaGetErrorString(status));
  }
}

void cuda_normalize_responsabilities(real * d_respons, int N_slices, int N_images){
  int nblocks = N_images;
  int nthreads = 256;
  cuda_normalize_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons, N_slices, N_images);
  /*
  int nblocks = N_slices;
  int nthreads = 256;
  cuda_normalize_responsabilities_uniform_kernel<<<nblocks,nthreads>>>(d_respons, N_slices, N_images);
  */
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
  }
}

__global__ void collapse_responsabilities_kernel(real *respons, int N_slices) {
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

void cuda_collapse_responsabilities(real *d_respons, int N_slices, int N_images) {
  int nblocks = N_images;
  int nthreads = 256;
  collapse_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons, N_slices);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (norm resp): %s\n",cudaGetErrorString(status));
  }
}

// x_log_x<T> computes the f(x) -> x*log(x)
template <typename T>
struct x_log_x
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

real cuda_total_respons(real * d_respons, real * respons,int n){
  thrust::device_ptr<real> p(d_respons);
  x_log_x<real> unary_op;
  thrust::plus<real> binary_op;
  real init = 0;
  // Calculates sum_0^n d_respons*log(d_respons)
  return thrust::transform_reduce(p, p+n, unary_op, init, binary_op);
}

void cuda_copy_slice_chunk_to_host(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d){
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  cudaMemcpy(&slices[slice_start],d_slices,sizeof(real)*N_2d*slice_chunk,cudaMemcpyDeviceToHost);

  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda copy slice to host time = %fms\n",k_ms);

}

void cuda_copy_slice_chunk_to_device(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d){
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  cudaMemcpy(d_slices,&slices[slice_start],sizeof(real)*N_2d*slice_chunk,cudaMemcpyHostToDevice);

  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  //printf("cuda copy slice to device time = %fms\n",k_ms);

}

void cuda_calculate_fit(real * slices, real * d_images, int * d_masks,
			real * d_scaling, real * d_respons, real * d_fit, real sigma,
			int N_2d, int N_images, int slice_start, int slice_chunk){
  //call the kernel  
  dim3 nblocks(N_images,slice_chunk);
  int nthreads = 256;
  calculate_fit_kernel<<<nblocks,nthreads>>>(slices, d_images, d_masks,
					     d_respons, d_fit, sigma, d_scaling,
					     N_2d, slice_start);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
  }
}

void cuda_calculate_fit_best_rot(real *slices, real * d_images, int *d_masks,
				 real *d_scaling, int *d_best_rot, real *d_fit,
				 int N_2d, int N_images, int slice_start, int slice_chunk) {
  dim3 nblocks(N_images, slice_chunk);
  int nthreads = 256;
  calculate_fit_best_rot_kernel<<<nblocks, nthreads>>>(slices, d_images, d_masks,
						       d_best_rot, d_fit, d_scaling,
						       N_2d, slice_start);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (fit): %s\n",cudaGetErrorString(status));
  }
}


void cuda_calculate_radial_fit(real *slices, real *d_images, int *d_masks,
			       real *d_scaling, real *d_respons, real *d_radial_fit,
			       real *d_radial_fit_weight, real *d_radius,
			       int N_2d, int side, int N_images, int slice_start,
			       int slice_chunk){
  dim3 nblocks(N_images,slice_chunk);
  int nthreads = 256;
  calculate_radial_fit_kernel<<<nblocks,nthreads>>>(slices, d_images, d_masks,
						    d_respons, d_scaling, d_radial_fit,
						    d_radial_fit_weight, d_radius,
						    N_2d, side, slice_start);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess) {
    printf("CUDA Error (radial fit): %s\n",cudaGetErrorString(status));
  }
}

void cuda_calculate_best_rotation(real *d_respons, int *d_best_rotation, int N_images, int N_slices){
  int nblocks = N_images;
  int nthreads = 256;
  calculate_best_rotation_kernel<<<nblocks, nthreads>>>(d_respons, d_best_rotation, N_slices);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    printf("CUDA Error (best rotation): %s\n", cudaGetErrorString(status));
  }
}

__global__ void multiply_by_gaussian_kernel(cufftComplex *model, const real sigma) {
  const int tid = threadIdx.x;
  const int step = blockDim.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y;
  const int model_side = gridDim.x;
  
  real radius2;
  real sigma2 = pow(sigma/(real)model_side, 2);
  int dx, dy, dz;
  if (model_side - y < y) {
    dy = model_side - y;
  } else {
    dy = y;
  }
  if (model_side - z < z) {
    dz = model_side - z;
  } else {
    dz = z;
  }

  for (int x = tid; x < (model_side/2+1); x += step) {
    if (model_side - x < x) {
      dx = model_side - x;
    } else { 
      dx = x;
    }

    // find distance to top left
    radius2 = pow((real)dx, 2) + pow((real)dy, 2) + pow((real)dz, 2);
    // calculate gaussian kernel

    /*
    model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2/((real)model_side))/(pow((real)model_side, 3));
    model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2/((real)model_side))/(pow((real)model_side, 3));
    */

    model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
    model[z*model_side*(model_side/2+1) + y*(model_side/2+1) + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));

    /*
    model[z*model_side*model_side + y*model_side + x].x *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
    model[z*model_side*model_side + y*model_side + x].y *= exp(-2.*pow(M_PI, 2)*radius2*sigma2)/(pow((real)model_side, 3));
    */
  }
}

__global__ void get_mask_from_model(real *model, int *mask, int size) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < size) {
    if (model[i] < 0.) {
      mask[i] = 0;
      model[i] = 0.;
    } else {
      mask[i] = 1;
    }
  }
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
  
  cudaFree(d_mask);
  cudaFree(ft);
 

}

/* Allocates and sets all weights to 1. */
void cuda_allocate_weight_map(real **d_weight_map, int image_side) {
  printf("allocating %d\n", image_side*image_side*sizeof(real));
  cudaMalloc(d_weight_map, image_side*image_side*sizeof(real));
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error (cuda_allocate_weight_map: copy): %s\n",cudaGetErrorString(status));
  }
  thrust::device_ptr<real> p(*d_weight_map);
  thrust::fill(p, p+image_side*image_side, real(1));
}

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
