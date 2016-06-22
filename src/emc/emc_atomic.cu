#include "emc.h"


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

__device__ void cuda_insert_slice(real *model, real *weight, real *slice,
				  int * mask, real w, real *rot, real *x_coordinates,
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
      //if (mask[y*x_max + x] == 1) {
      if (mask[y*x_max + x] == 1 && slice[y*x_max + x] >= 0.0) {
	/* This is just a matrix multiplication with rot */
	new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x];
	new_y =	m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x];
	new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x];
	/* changed the next lines +0.5 -> -0.5 (11 dec 2012)*/
	round_x = lroundf(model_x/2.0f - 0.5f + new_x);
	round_y = lroundf(model_y/2.0f - 0.5f + new_y);
	round_z = lroundf(model_z/2.0f - 0.5f + new_z);
	if (round_x >= 0 && round_x < model_x &&
	    round_y >= 0 && round_y < model_y &&
	    round_z >= 0 && round_z < model_z) {
	  /* this is a simple compile time check that can go bad at runtime, but such is life */
#if __CUDA_ARCH__ >= 200
	  atomicAdd(&model[round_z*model_x*model_y + round_y*model_x + round_x], w * slice[y*x_max + x]);
	  atomicAdd(&weight[round_z*model_x*model_y + round_y*model_x + round_x], w);
#else
	  atomicFloatAdd(&model[round_z*model_x*model_y + round_y*model_x + round_x], w * slice[y*x_max + x]);
	  atomicFloatAdd(&weight[round_z*model_x*model_y + round_y*model_x + round_x], w);
#endif
	  //	  model[(round_z*model_x*model_y + round_y*model_x + round_x)] += w * slice[y*x_max + x];	    
	  //	  weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;
	}
      }
    }
  }
}

__device__ void interpolate_model_set(real *model, real *model_weight, int model_x, int model_y, int model_z,
				      real new_x, real new_y, real new_z, real value, real value_weight) {
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

#if __CUDA_ARCH__ >= 200
	  atomicAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value*value_weight);
	  atomicAdd(&model_weight[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value_weight);
#else
	  atomicFloatAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value*value_weight);
	  atomicFloatAdd(&model_weight[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value_weight);
#endif
	}
      }
    }
  }
}

__device__ void cuda_insert_slice_interpolate(real *model, real *weight, real *slice,
					      int * mask, real w, real *rot, real *x_coordinates,
					      real *y_coordinates, real *z_coordinates, int slice_rows,
					      int slice_cols, int model_x, int model_y, int model_z,
					      int tid, int step)
{
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
      //if (mask[y*x_max + x] == 1) {
      if (slice[y*x_max + x] >= 0.0) {
	/* This is just a matrix multiplication with rot */
	new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x] + model_x/2.0 - 0.5;
	new_y =	m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x] + model_y/2.0 - 0.5;
	new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x] + model_z/2.0 - 0.5;
	
	interpolate_model_set(model, weight, model_x, model_y, model_z, new_x, new_y, new_z, slice[y*x_max + x], w);
      }
    }
  }
}

__device__ void cuda_insert_slice_final_interpolate(real *model, real *weight, real *slice,
						    int * mask, real w, real *rot, real *x_coordinates,
						    real *y_coordinates, real *z_coordinates, int slice_rows,
						    int slice_cols, int model_x, int model_y, int model_z,
						    int tid, int step)
{
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
      //if (mask[y*x_max + x] == 1) {
      if (slice[y*x_max + x] >= 0.0) {
	/* This is just a matrix multiplication with rot */
	new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x] + model_x/2.0 - 0.5;
	new_y =	m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x] + model_y/2.0 - 0.5;
	new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x] + model_z/2.0 - 0.5;
	
	interpolate_model_set(model, weight, model_x, model_y, model_z, new_x, new_y, new_z, slice[y*x_max + x], w);
      }
    }
  }
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
  //model[side*side*4 + side*4 + 4] = -1.;
  //model[side*side*5 + side*4 + 4] = -1.;
  //model[side*side*4 + side*4 + 5] = -1.;
  //model[side*side*4 + side*5 + 5] = -1.;

  real interp_x = 3.4;
  real interp_y = 4.6;
  real interp_z = 3.5;
  real value = 1.;
  real value_weight = 1.;
  interpolate_model_set(model, weight, side, side, side, interp_x, interp_y, interp_z, value, value_weight);
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
  /*
  printf("weight\n");
  for (int z = 0; z <= 4; z++) {
    for (int y = 0; y <= 4; y++) {
      for (int x = 0; x <= 4; x++) {
	printf("%g ", weight[side*side*z + side*y + x]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */
  printf("test interpolation end\n");
}

__global__ void update_slices_kernel(real * images, real * slices, int * masks, real * respons,
				     real * scaling, int * active_images, int N_images, int slice_start, int N_2d,
				     real * slices_total_respons, real * rot,
				     real * x_coord, real * y_coord, real * z_coord,
				     real * model,
				     int slice_rows, int slice_cols,
				     int model_x, int model_y, int model_z){
  /* each block takes care of 1 slice */
  int i_slice = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  real total_respons = 0.0f;

  for (int i = tid; i < N_2d; i+=step) {
    real sum = 0;
    for (int i_image = 0; i_image < N_images; i_image++) {
      if (active_images[i_image] > 0 && masks[i_image*N_2d+i] != 0) {
	sum += images[i_image*N_2d+i]*
	  respons[(slice_start+i_slice)*N_images+i_image]/scaling[(slice_start+i_slice)*N_images+i_image];
      }
    }
    slices[i_slice*N_2d+i] = sum;
    if (slices[i_slice*N_2d+i] <= 0.) {
      slices[i_slice*N_2d+i] = -1.0;
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
      if (slices[i_slice*N_2d+i] >= 0.) {
	slices[i_slice*N_2d+i] /= total_respons;
      }
    }
  }
}

__global__ void update_slices_final_kernel(real * images, real * slices, int * masks, real * respons,
					   real * scaling, int * active_images, int N_images,
					   int slice_start, int N_2d,
					   real * slices_total_respons, real * rot,
					   real * x_coord, real * y_coord, real * z_coord,
					   real * model, real * weight,
					   int slice_rows, int slice_cols,
					   int model_x, int model_y, int model_z){
  /* each block takes care of 1 slice */
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
  /*
  if(total_respons > 1e-10f){
    for (int i = tid; i < N_2d; i+=step) {
      slices[i_slice*N_2d+i] /= total_respons;
    }
  } else {
    slices[i_slice*N_2d+i] = -1.0;
  }
  */
}

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
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
  if(total_respons > 1e-10f){
    /*
    cuda_insert_slice(model, weight, &slices[i_slice*N_2d], mask, total_respons,
		      &rot[4*i_slice], x_coord, y_coord, z_coord,
		      slice_rows, slice_cols, model_x, model_y, model_z, tid, step);
    */

    cuda_insert_slice_interpolate(model, weight, &slices[i_slice*N_2d], mask, total_respons,
				  &rot[4*i_slice], x_coord, y_coord, z_coord,
				  slice_rows, slice_cols, model_x, model_y, model_z, tid, step);

  }
}


__global__ void insert_slices_final_kernel(real * images, real * slices, int * mask, real * respons,
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
  if(total_respons > 1e-10f){
    /*
    cuda_insert_slice(model, weight, &slices[i_slice*N_2d], mask, total_respons,
		      &rot[4*i_slice], x_coord, y_coord, z_coord,
		      slice_rows, slice_cols, model_x, model_y, model_z, tid, step);
    */

    cuda_insert_slice_final_interpolate(model, weight, &slices[i_slice*N_2d], mask, total_respons,
					&rot[4*i_slice], x_coord, y_coord, z_coord,
					slice_rows, slice_cols, model_x, model_y, model_z, tid, step);

  }
}

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

__device__ void cuda_calculate_responsability_absolute_atomic(float *slice, float *image, int *mask, real sigma, real scaling, int N_2d, int tid, int step, real * sum_cache, int * count_cache)
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
  //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_responsability_poisson_atomic(float *slice, float *image,
							     int *mask, real sigma, real scaling,
							     int N_2d, int tid, int step,
							     real * sum_cache, int * count_cache)
{
  real sum = 0.0;
  const int i_max = N_2d;
  int count = 0;
  for (int i = tid; i < i_max; i+=step) {
    if (mask[i] != 0 && slice[i] >= 0.0f) {
      //sum += pow((slice[i] - image[i]/scaling) / (sqrt(image[i])+0.4), 2);
      sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]+0.02), 2); // 0.2 worked
      //sum += pow((slice[i] - image[i]/scaling) / sqrt(image[i]/0.5+10.0), 2); // 0.2 worked
      //sum += pow((slice[i]*scaling - image[i])/8.0/ (sqrt(image[i]/8.0 + 1.0)), 2); // 0.2 worked
      count++;
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count;
  //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_responsability_true_poisson_atomic(float *slice, float *image,
							   int *mask, real sigma, real scaling,
							   int N_2d, int tid, int step,
							   real * sum_cache, int * count_cache)
{
  real sum = 0.0;
  const int i_max = N_2d;
  int count = 0;
  for (int i = tid; i < i_max; i+=step) {
    if (mask[i] != 0 && slice[i] >= 0.0f) {
      sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+0.2), 2); // 0.2 worked
      //sum += pow(slice[i]*scaling - image[i], 2) / ((image[i]/8.0 + 1.0)*pow(8.0, 2));
      //sum += pow((slice[i] - image[i]/scaling) / sqrt(slice[i]+1.0), 2);
      count++;
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count;
  //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__device__ void cuda_calculate_fit_error(float* slice, float *image,
				   int *mask, real scaling,
				   int N_2d, int tid, int step,
				   real *sum_cache, int *count_cache) {
  real sum = 0.0;
  const int i_max = N_2d;
  int count = 0;
  for (int i = tid; i < i_max; i+=step) {
    //if (mask[i] != 0 && slice[i] >= 0.0f) {
    if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
      sum += abs((slice[i] - image[i]/scaling) / (slice[i] + image[i]/scaling));
      count++;
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count;
}

/* i should try using a correlation coefficient at some point. The following will do a (almost) Fourier correlation where the phase is offcourse missing. Needs a function to collrect the result still. */
/*
__device__ void cuda_calculate_fit2_correlation(float *slice, float *image, int *mask, real scaling,
						int N_2d, int tid, int step, real *nom_cache, real *den_cache) {
  real nom = 0.;
  real den1 = 0.;
  real den2 = 0.;
  const int i_max = N_2d;
  for (int i = tid; i < i_max; i+=step) {
    if (mask[i] != 0 && slice[i] > 0.f && image[i] > 0.f) {
      nom += slice[i]*image[i];
      den1 += slice[i]**2;
      den2 += slice[i]**2;
    }
  }
  nom_cache[tid] = nom;
  den1_cache[tid] = den1;
  den2_cache[tid] = den2;
}
*/

__device__ void cuda_calculate_fit2_error(float* slice, float *image,
					  int *mask, real scaling,
					  int N_2d, int tid, int step,
					  real *nom_cache, real *den_cache) {
  real nom = 0.0;
  real den = 0.0;
  const int i_max = N_2d;
  for (int i = tid; i < i_max; i+=step) {
    //if (mask[i] != 0 && slice[i] >= 0.0f) {
    if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
      nom += abs(slice[i] - image[i]/scaling);
      den += abs(slice[i] + image[i]/scaling);
    }
  }
  nom_cache[tid] = nom;
  den_cache[tid] = den;
}


__device__ real cuda_calculate_single_fit_error(float* slice, float *image,
						int *mask, real scaling,
						int N_2d, int tid, int step) {
  __shared__ real sum_cache[256];
  __shared__ int count_cache[256];
  real sum = 0.0;
  const int i_max = N_2d;
  int count = 0;
  for (int i = tid; i < i_max; i+=step) {
    //if (mask[i] != 0 && slice[i] >= 0.0f) {
    if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
      sum += abs((slice[i] - image[i]/scaling) / (slice[i] + image[i]/scaling));
      count++;
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count;
  inblock_reduce(sum_cache);
  inblock_reduce(count_cache);
  return sum_cache[0]/count_cache[0];
}

__device__ real cuda_calculate_single_fit2_error(float* slice, float *image,
						int *mask, real scaling,
						int N_2d, int tid, int step) {
  __shared__ real nom_cache[256];
  __shared__ real den_cache[256];
  real nom = 0.0;
  real den = 0.0;
  const int i_max = N_2d;
  for (int i = tid; i < i_max; i+=step) {
    //if (mask[i] != 0 && slice[i] >= 0.0f) {
    if (mask[i] != 0 && slice[i] > 0.0f && image[i] >0.0f) {
      nom += abs(slice[i] - image[i]/scaling);
      den += abs(slice[i] + image[i]/scaling);
    }
  }
  nom_cache[tid] = nom;
  den_cache[tid] = den;
  inblock_reduce(nom_cache);
  inblock_reduce(den_cache);

  return nom_cache[0]/(den_cache[0]);
}



__global__ void calculate_fit_kernel(real *slices, real *images, int *masks,
				     real *respons, real *fit, real sigma,
				     real *scaling, int N_2d, int slice_start){
  __shared__ real sum_cache[256];
  __shared__ int count_cache[256];
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;

  cuda_calculate_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[i_image*N_2d],
			   scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step, sum_cache, count_cache);

  inblock_reduce(sum_cache);
  inblock_reduce(count_cache);
  __syncthreads();
  if (tid == 0) {
    atomicAdd(&fit[i_image], sum_cache[0]/count_cache[0]*respons[(slice_start+i_slice)*N_images+i_image]);
  }
}

__global__ void calculate_fit2_kernel(real *slices, real *images, int *masks,
				     real *respons, real *fit, real sigma,
				     real *scaling, int N_2d, int slice_start){
  //__shared__ real sum_cache[256];
  __shared__ real nom_cache[256];
  __shared__ real den_cache[256];
  //__shared__ int count_cache[256];
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;

  cuda_calculate_fit2_error(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[i_image*N_2d],
			    scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step, nom_cache, den_cache);

  inblock_reduce(nom_cache);
  inblock_reduce(den_cache);
  __syncthreads();
  if (tid == 0) {
    atomicAdd(&fit[i_image], nom_cache[0]/(den_cache[0])*respons[(slice_start+i_slice)*N_images+i_image]);
  }
}

__global__ void calculate_fit_best_rot_kernel(real *slices, real *images, int *masks,
					      int *best_rot, real *fit,
					      real *scaling, int N_2d, int slice_start){
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;
  
  real this_fit;
  if (best_rot[i_image] == (slice_start+i_slice)) {

    this_fit = cuda_calculate_single_fit_error(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[i_image*N_2d],
					       scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);
    
    if (tid == 0) {
      fit[i_image] = this_fit;
    }
  }
}

__global__ void calculate_fit_best_rot2_kernel(real *slices, real *images, int *masks,
					      int *best_rot, real *fit,
					      real *scaling, int N_2d, int slice_start){
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;
  
  real this_fit;
  if (best_rot[i_image] == (slice_start+i_slice)) {
    this_fit = cuda_calculate_single_fit2_error(&slices[i_slice*N_2d], &images[i_image*N_2d], &masks[i_image*N_2d],
					       scaling[(slice_start+i_slice)*N_images+i_image], N_2d, tid, step);
    
    if (tid == 0) {
      fit[i_image] = this_fit;
    }
  }
}

  /* calcualte the fit as a function of radius */
__global__ void calculate_radial_fit_kernel(real * slices , real * images, int * masks,
					    real * respons, real * scaling, real * radial_fit,
					    real * radial_fit_weight, real * radius,
					    int N_2d,  int side,  int slice_start){
  __shared__ real sum_cache[256]; //256
  __shared__ real weight_cache[256];
  const int max_radius = side/2;
  //printf("before alloc\n");
  //printf("after alloc\n");
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;
  real error;
  int rad;

  if (tid < max_radius) {
    sum_cache[tid] = 0.0;
    weight_cache[tid] = 0.0;
  }

  __syncthreads();
  real this_resp = respons[(slice_start+i_slice)*N_images+i_image];
  //printf("%g\n", this_resp);
  //if (this_resp > 1.0e-10) {
  for (int i = tid; i < N_2d; i += step) {
    if (masks[i_image*N_2d+i] != 0 && slices[i_slice*N_2d+i] > 0.0f) {
      error = fabs((slices[i_slice*N_2d+i] - images[i_image*N_2d+i]/scaling[(slice_start+i_slice)*N_images+i_image]) / 
		   (slices[i_slice*N_2d+i] + images[i_image*N_2d+i]/scaling[(slice_start+i_slice)*N_images+i_image]));
      /*
      if (i_image == 0 && i_slice == 0) {
	printf("error[i_slice=%d, i_image=%d, i=%d] = %g, (%g ~ %g) (resp = %g)\n", i_slice, i_image, i, error, slices[i_slice*N_2d+i], images[i_image*N_2d+i]/scaling[i_image], this_resp);
      }
      */
      rad = (int)radius[i];

      if (rad < max_radius) {
	atomicAdd(&sum_cache[rad],error*this_resp);
	atomicAdd(&weight_cache[rad],this_resp);
      }
    }
  }
    //}
  __syncthreads();
  if (tid < max_radius) {
    atomicAdd(&radial_fit[tid],sum_cache[tid]);
    atomicAdd(&radial_fit_weight[tid],weight_cache[tid]);
  }
}

/* 
__global__ void calulate_standard_deviation(real *slices, real *images, int *mask, real *respons, real *scaling,
					    real *radial_std, real *radius, int N_2d, int side, int slice_start) {
  __shared__ real nom[256];
  __shared__ real den[256];
  nom[tid] = 0.;
  den[tid] = 0.;

  for (int i = tid; i < N_2d; i += step) {
    nom[tid] += 
  }
}
*/
