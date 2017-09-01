/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */
#ifndef EMC_MATH_H
#define EMC_MATH_H
#include <emc_common.h>
#include "errors.h"


int compare_real(const void *pa, const void *pb);
void calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength,
                           sp_matrix *x_coordinates,
                           sp_matrix *y_coordinates, sp_matrix *z_coordinates);
void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
                  sp_imatrix * mask, real w, Quaternion rot, sp_matrix *x_coordinates,
                  sp_matrix *y_coordinates, sp_matrix *z_coordinates);
void normalize_images(sp_matrix **images, sp_imatrix *mask, const Configuration conf, real central_part_radius);
void normalize_images_central_part(sp_matrix ** const images, const sp_imatrix * const mask, real radius, const Configuration conf) ;
void normalize_images_individual_mask(sp_matrix **images, sp_imatrix **masks,
                                      Configuration conf);
void normalize_images_preserve_scaling(sp_matrix ** images, sp_imatrix *mask, Configuration conf);
void create_initial_model_uniform(sp_3matrix *model, gsl_rng *rng);
void create_initial_model_radial_average(sp_3matrix *model, sp_matrix **images,  int N_images,
                        sp_imatrix *mask, real initial_model_noise,
                        gsl_rng *rng);
void create_initial_model_random_orientations(sp_3matrix *model, sp_3matrix *weight, sp_matrix **images,
                                                      int N_images, sp_imatrix *mask, sp_matrix *x_coordinates,
                                                     sp_matrix *y_coordinates, sp_matrix *z_coordinates, gsl_rng *rng);
void create_initial_model_given_orientations(sp_3matrix *model, sp_3matrix *weight,
                                                    sp_matrix **images, int  N_images,
                                                    sp_imatrix *mask, sp_matrix *x_coordinates,
                                                    sp_matrix *y_coordinates,sp_matrix *z_coordinates, const char *init_rotations_file);
void create_initial_model_file(sp_3matrix *model, const char *model_file);

void model_init(Configuration , sp_3matrix *,
                sp_3matrix * , sp_matrix ** , sp_imatrix *,
                sp_matrix *, sp_matrix *, sp_matrix *);


//added by Jing 2016-10-25
int get_allocate_len(int ntasks, int N_slices, int taskid);
void copy_real(int len, real* source , real* dst);
void sum_vectors(real* ori, real* tmp, int len);
void max_max_vector(real* a, real* b, int len);
void log_vector(real* a, int len);
double real_real_validate(real* a, real*b , int len);
void minus_vector (real* sub_out, real* min, int len);
void set_vector(real* dst, real* ori, int len);
bool vector_range (real low, real high,int len, real* vector);
void min_and_max(real* vector, int len, real* min, real* max);
int  get_max_allocate_len(int ntasks, int N_slices);
void reset_to_zero(void*, int, size_t);
void devide_part(real* data, int len, int part);
real calcualte_image_max(sp_imatrix * mask, sp_matrix** images, int N_images, int N_2d);
int calculate_image_included(const Configuration conf, int N_images);

unsigned long int get_seed(Configuration conf);
void calculate_distance_spmatrix(sp_matrix* radius, Configuration conf);
void set_int_array(int * array, int value, int len);
void set_real_array(real * array, real value, int len);

void compute_best_rotations(real* full_respons, int N_images, int N_slices, int* best_rotation);
void compute_best_respons(real* respons, int N_images, int N_slices, real* best_respons);
#endif
