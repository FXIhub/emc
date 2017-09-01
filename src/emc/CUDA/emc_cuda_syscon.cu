/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda.h>
#include<emc_cuda_syscon_host.h>

/*
#ifdef __cplusplus 
extern "C" {
#endif
*/
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

  //qsort(core_count_pointers, N_devices, sizeof(core_count_pointers[0]), compare);
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
}
/*
#ifdef __cplusplus 
}
#endif
*/
