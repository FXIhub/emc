/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda.h>


void cuda_allocate_slices(real ** slices, int side, int N_slices){
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
    int start = 0;
    printf("debug %d %f %f %f %f \n\n",start,  (rotations[start]),
           (rotations[start+1]), (rotations[start+2]), (rotations[start+3]));
    start = 25050;
    printf("debug %d %f %f %f %f \n\n",start,  (rotations[start]),
           (rotations[start+1]), (rotations[start+2]), (rotations[start+3]));
    cudaMemcpy(*d_rotations, rotations, sizeof(real)*4*N_slices, cudaMemcpyHostToDevice);
    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_rotations: copy): %s\n",cudaGetErrorString(status));
    }
}

void cuda_allocate_rotations_chunk(real ** d_rotations, Quaternion * rotations, int start, int end){
    cudaMalloc(d_rotations,sizeof(real)*4*(end-start));
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_rotations_chunk: malloc): %s\n",cudaGetErrorString(status));
    }

    printf("debug %d %f %f %f %f \n\n",start,  (rotations[start]),
           (rotations[start+1]), (rotations[start+2]), (rotations[start+3]));

    cudaMemcpy(*d_rotations,&(rotations[start]),sizeof(real)*4 *(end-start),cudaMemcpyHostToDevice);
    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_rotations_chunk: copy): %s\n",cudaGetErrorString(status));
    }
}

void cuda_copy_rotations_chunk(real ** d_rotations, Quaternion * rotations, int start, int end){
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_rotations_chunk: malloc): %s\n",cudaGetErrorString(status));
    }

    printf("debug %d %f %f %f %f \n\n",start,  (rotations[start]),
           (rotations[start+1]), (rotations[start+2]), (rotations[start+3]));

    cudaMemcpy(*d_rotations,&(rotations[start]),sizeof(real)*4 *(end-start),cudaMemcpyHostToDevice);
    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_rotations_chunk: copy): %s\n",cudaGetErrorString(status));
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

void cuda_allocate_masks(int ** d_images, sp_imatrix ** images,  int N_images){

    cudaMalloc(d_images,sizeof(int)*sp_imatrix_size(images[0])*N_images);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_masks: malloc): %s\n",cudaGetErrorString(status));
    }
    for(int i = 0;i<N_images;i++){
        cudaMemcpy(&(*d_images)[sp_imatrix_size(images[0])*i],images[i]->data,sizeof(int)*sp_imatrix_size(images[0]),cudaMemcpyHostToDevice);
    }
    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_masks: copy): %s\n",cudaGetErrorString(status));
    }
}

void cuda_reset_model(sp_3matrix * model, real * d_model){
    cudaMemset(d_model,0,sizeof(real)*sp_3matrix_size(model));
}

void cuda_copy_model(sp_3matrix * model, real *d_model){
    cudaMemcpy(model->data,d_model,sizeof(real)*sp_3matrix_size(model),cudaMemcpyDeviceToHost);
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
void cuda_copy_real(real *dst, real *src, int n){
    cudaMemcpy(dst,src,n*sizeof(real),cudaMemcpyDeviceToDevice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_real): %s\n",cudaGetErrorString(status));
    }
}

void cuda_copy_real_to_host(real *x, real *d_x, int n){
    cudaMemcpy(x,d_x,n*sizeof(real),cudaMemcpyDeviceToHost);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_real_to_host): %s\n",cudaGetErrorString(status));
    }
}

void cuda_copy_int_to_device(int *x, int *d_x, int n){
    cudaMemcpy(d_x,x,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_int_to_device): %s\n",cudaGetErrorString(status));
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
    status = cudaGetLastError();
   if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_scaling): %s\n",cudaGetErrorString(status));
    }
}

void cuda_allocate_scaling_full(real **d_scaling, int N_images, int N_slices) {
    cudaMalloc(d_scaling, N_images*N_slices*sizeof(real));
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_scaling_full): %s\n",cudaGetErrorString(status));
    }
    thrust::device_ptr<real> p(*d_scaling);
    thrust::fill(p, p+N_images*N_slices, real(1.));
    status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_scaling_full): %s\n",cudaGetErrorString(status));
    }
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
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_slice_chunk_to_host): %s\n",cudaGetErrorString(status));
    }
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
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_slice_chunk_to_device): %s\n",cudaGetErrorString(status));
    }
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

/* Allocates and sets all weights to 1. */
void cuda_allocate_weight_map(real **d_weight_map, int image_side) {
    cudaMalloc(d_weight_map, image_side*image_side*sizeof(real));
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_allocate_weight_map: copy): %s\n",cudaGetErrorString(status));
    }
    thrust::device_ptr<real> p(*d_weight_map);
    thrust::fill(p, p+image_side*image_side, real(1));
}

void cuda_set_real_array(real **d_array, int n, real value) {
    thrust::device_ptr<real> p(*d_array);
    thrust::fill(p, p+n, value);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_set_real_array: copy): %s\n",cudaGetErrorString(status));
    }

}

void cuda_copy_weight_to_device(real *x, real *d_x, int n, int taskid){
    int y=taskid * n;
    cudaMemcpy(d_x,&(x[y]),n*sizeof(real),cudaMemcpyHostToDevice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_weight_to_device): %s\n",cudaGetErrorString(status));
    }
}

void cuda_reset_real(real *d_real, int len){
    cudaMemset(d_real,0,sizeof(real)*len);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_reset_real): %s\n",cudaGetErrorString(status));
    }
}


void cuda_mem_free(real * d){
    cudaError_t status = cudaFree(d);
    if(status != cudaSuccess){
        printf("CUDA Error (mem free): %s\n",cudaGetErrorString(status));
    }
}
void cuda_copy_model_2_device (real ** d_model, sp_3matrix * model){
    cudaMemcpy(*d_model,model->data,sizeof(real)*sp_3matrix_size(model),cudaMemcpyHostToDevice);
    cudaError_t status = cudaGetLastError();
    if(status != cudaSuccess){
        printf("CUDA Error (cuda_copy_model_2_device): %s\n",cudaGetErrorString(status));
    }
}


