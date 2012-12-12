#pragma once 

#include <spimage.h>
#include "rotations.h"

#ifdef __cplusplus
extern "C"{
#endif

  enum diff_type {absolute=0, poisson, relative};

typedef struct{
  int side;
  real wavelength;
  real pixel_size;
  int detector_size;
  real detector_distance;
}Setup;

  typedef struct{
    int model_side;
    int read_stride;
    double wavelength;
    double pixel_size;
    int detector_size;
    double detector_distance;
    int rotations_n;
    double sigma;
    int slice_chunk;
    int N_images;
    int max_iterations;
    int blur_image;
    double blur_sigma;
    const char *mask_file;
    const char *image_prefix;
    int normalize_images;
    int known_intensity;
    int model_input;
    const char *model_file;
    int exclude_images;
    double exclude_ratio;
  }Configuration;

  void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
		    sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
		    sp_matrix *y_coordinates, sp_matrix *z_coordinates);

  Configuration read_configuration_file(const char *filename);

  void cuda_update_slices(real * d_images, real * slices, int * d_mask,
			  real * d_respons, real * d_scaling, int * d_active_images, int N_images,
			  int slice_start, int slice_chunk, int N_2d,
			  sp_3matrix * model, real * d_model, 
			  real  *x_coordinates, real *y_coordinates,
			  real *z_coordinates, real *d_rotations,
			  real * d_weight, sp_matrix ** images);
  void cuda_update_slices_final(real * d_images, real * d_slices, int * d_mask,
			  real * d_respons, real * d_scaling, int * d_active_images, int N_images,
			  int slice_start, int slice_chunk, int N_2d,
			  sp_3matrix * model, real * d_model, 
			  real  *x_coordinates, real *y_coordinates,
			  real *z_coordinates, real *d_rotations,
			  real * d_weight,sp_matrix ** images);

  void cuda_update_weighted_power(real * d_images, real * d_slices, int * d_mask,
				  real * d_respons, real * d_weighted_power, int N_images,
				  int slice_start, int slice_chunk, int N_2d);

  void cuda_update_scaling(real * d_images, int * d_mask,
			   real * d_scaling, real * d_weighted_power,
			   int N_images, int N_slices, int N_2d, real * scaling);
  void cuda_update_scaling_best(real *d_images, int *d_mask,
				real *d_model, real *d_scaling, real *d_respons, real *d_rotations,
				real *x_coordinates, real *y_coordinates, real *z_coordinates,
				int N_images, int N_slices, int side, real *scaling);
  void cuda_update_scaling_full(real *d_images, real *d_slices, int *d_mask, real *d_scaling,
				int N_2d, int N_images, int slice_start, int slice_chunk, enum diff_type diff);


  void cuda_calculate_responsabilities(real * d_slices, real * d_images, int * d_mask,
				       real sigma, real * d_scaling, real * d_respons, real *d_weights, 
				       int N_2d, int N_images, int slice_start, int slice_chunk,
				       enum diff_type diff);
  void cuda_calculate_responsabilities_sum(real * respons, real * d_respons, int N_slices,
					   int N_images);

  void cuda_get_slices(sp_3matrix * model, real * d_model, real * d_slices, real * d_rot, 
		       real * d_x_coordinates, real * d_y_coordinates,
		       real * d_z_coordinates, int start_slice, int slice_chunk);

  void cuda_allocate_slices(real ** slices, int side, int N_slices);
  real cuda_model_max(real * model, int model_size);
  real cuda_model_average(real * model, int model_size);
  void cuda_allocate_model(real ** d_model, sp_3matrix * model);
  void cuda_allocate_mask(int ** d_mask, sp_imatrix * mask);
  void cuda_reset_model(sp_3matrix * model, real * d_model);
  void cuda_copy_model(sp_3matrix * model, real *d_model);
  void cuda_divide_model_by_weight(sp_3matrix * model, real * d_model, real * d_weight);
  void cuda_normalize_model(sp_3matrix * model, real * d_model);
  void cuda_allocate_rotations(real ** d_rotations, Quaternion ** rotations, int N_slices);
  void cuda_allocate_images(real ** d_images, sp_matrix ** images, int N_images);
  void cuda_allocate_masks(int ** d_images, sp_imatrix ** images,  int N_images);
  void cuda_allocate_coords(real ** d_x, real ** d_y, real ** d_z, sp_matrix * x,
			  sp_matrix * y, sp_matrix * z);
  void cuda_allocate_real(real ** x, int n);
  void cuda_allocate_int(int ** x, int n);
  void cuda_allocate_scaling(real ** scaling, int N_images);
  void cuda_allocate_scaling_full(real ** scaling, int N_images, int N_slices);
  void cuda_normalize_responsabilities(real * d_respons, int N_slices, int N_images);
  void cuda_normalize_responsabilities_single(real *d_respons, int N_slices, int N_images);
  real cuda_total_respons(real * d_respons, real * respons, int n);
  void cuda_set_to_zero(real * x, int n);
  void cuda_copy_real_to_device(real *x, real *d_x, int n);
  void cuda_copy_real_to_host(real *x, real *d_x, int n);
  void cuda_copy_int_to_device(int *x, int *d_x, int n);
  void cuda_copy_int_to_host(int *x, int *d_x, int n);
  void cuda_copy_slice_chunk_to_host(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d);
  void cuda_copy_slice_chunk_to_device(real * slices, real * d_slices, int slice_start, int slice_chunk, int N_2d);
  void cuda_calculate_fit(real * slices, real * d_images, int * d_mask,
			  real * d_scaling, real * d_respons, real * d_fit, real sigma,
			  int N_2d, int N_images, int slice_start, int slice_chunk);
  void cuda_calculate_fit_best_rot(real *slices, real * d_images, int *d_mask,
				   real *d_scaling, int *d_best_rot, real *d_fit,
				   int N_2d, int N_images, int slice_start, int slice_chunk);
  void cuda_calculate_radial_fit(real *slices, real *d_images, int *d_mask,
				 real *d_scaling, real *d_respons, real *d_radial_fit,
				 real *d_radial_fit_weight, real *d_radius,
				 int N_2d, int side, int N_images, int slice_start,
				 int slice_chunk);
  void cuda_calculate_best_rotation(real *d_respons, int *d_best_rotation, int N_images, int N_slices);

#ifdef __cplusplus
  }
#endif
