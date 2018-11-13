#ifndef EMC_CUDA_COMMON_HOST_H
#define EMC_CUDA_COMMON_HOST_H

int compare(const void *a, const void *b);
void cuda_max_vector(real* d_matrix, int N_images, int N_slices, real* d_maxr);
void cuda_matrix_scalar(real* d_matrix, int N_images, int N_slices, real d_scalar);


#endif
