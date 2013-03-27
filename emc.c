//#include "fragmentation.h"
#include <spimage.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include "emc.h"
//#include "rotations.h"
#include <libconfig.h>
#include <signal.h>
#include <sys/stat.h>
#include <hdf5.h>

static int quit_requested = 0;

void nice_exit(int sig) {
  if (quit_requested == 0) {
    quit_requested = 1;
  } else {
    exit(1);
  }
}

int compare_real(const void *pa, const void *pb) {
  real a = *(const real*)pa;
  real b = *(const real*)pb;

  if (a < b) {return -1;}
  else if (a > b) {return 1;}
  else {return 0;}
}


void write_1d_array_hdf5(char *filename, real *array, int index1_max) {
  hid_t file_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t space_id;
  hid_t dataset_id;
  hsize_t dim[1]; dim[0] = index1_max;
  
  space_id = H5Screate_simple(1, dim, NULL);
  dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}

void write_2d_array_hdf5(char *filename, real *array, int index1_max, int index2_max) {
  hid_t file_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t space_id;
  hid_t dataset_id;
  hsize_t dim[2]; dim[0] = index1_max; dim[1] = index2_max;
  
  space_id = H5Screate_simple(2, dim, NULL);
  dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}

void write_2d_array_trans_hdf5(char *filename, real *array, int index1_max, int index2_max) {

  real *array_trans = malloc(index1_max*index2_max*sizeof(real));
  for (int i1 = 0; i1 < index1_max; i1++) {
    for (int i2 = 0; i2 < index2_max; i2++) {
      array_trans[i2*index1_max+i1] = array[i1*index2_max+i2];
    }
  }
  hid_t file_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t space_id;
  hid_t dataset_id;
  hsize_t dim[2]; dim[0] = index2_max; dim[1] = index1_max;
  
  space_id = H5Screate_simple(2, dim, NULL);
  dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array_trans);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
  free(array_trans);
}

void write_3d_array_hdf5(char *filename, real *array, int index1_max, int index2_max, int index3_max) {
  hid_t file_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t space_id;
  hid_t dataset_id;
  hsize_t dim[3];
  dim[0] = index1_max; dim[1] = index2_max; dim[2] = index3_max;
  
  space_id = H5Screate_simple(3, dim, NULL);
  dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}

void calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength,
			   sp_matrix *x_coordinates,
			   sp_matrix *y_coordinates, sp_matrix *z_coordinates) {
  const int x_max = side;
  const int y_max = side;
  real pixel_r, real_r, fourier_r, angle_r, fourier_z;
  real pixel_x, pixel_y, pixel_z;
  //tabulate angle later
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      pixel_r = sqrt(pow((real)(x-x_max/2)+0.5,2) + pow((real)(y-y_max/2)+0.5,2));
      real_r = pixel_r*pixel_size;
      angle_r = atan2(real_r,detector_distance);
      fourier_r = sin(angle_r)/wavelength;
      fourier_z = (1. - cos(angle_r))/wavelength;

      pixel_x = (real)(x-x_max/2)+0.5;
      pixel_y = (real)(y-y_max/2)+0.5;
      pixel_z = fourier_z/fourier_r*pixel_r;
      sp_matrix_set(x_coordinates,x,y,pixel_x);
      sp_matrix_set(y_coordinates,x,y,pixel_y);
      sp_matrix_set(z_coordinates,x,y,pixel_z);
    }
  }
}

void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
		  sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
		  sp_matrix *y_coordinates, sp_matrix *z_coordinates)
{
  const int x_max = sp_matrix_rows(slice);
  const int y_max = sp_matrix_cols(slice);
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      if (sp_imatrix_get(mask,x,y) == 1) {
	/* This is just a matrix multiplication with rot */
	new_x =
	  (rot->q[0]*rot->q[0] + rot->q[1]*rot->q[1] -
	   rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (2.0*rot->q[1]*rot->q[2] -
	   2.0*rot->q[0]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (2.0*rot->q[1]*rot->q[3] +
	   2.0*rot->q[0]*rot->q[2])*sp_matrix_get(z_coordinates,x,y);
	new_y =
	  (2.0*rot->q[1]*rot->q[2] +
	   2.0*rot->q[0]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] +
	   rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (2.0*rot->q[2]*rot->q[3] -
	   2.0*rot->q[0]*rot->q[1])*sp_matrix_get(z_coordinates,x,y);
	new_z =
	  (2.0*rot->q[1]*rot->q[3] -
	   2.0*rot->q[0]*rot->q[2])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (2.0*rot->q[2]*rot->q[3] +
	   2.0*rot->q[0]*rot->q[1])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] -
	   rot->q[2]*rot->q[2] + rot->q[3]*rot->q[3])*sp_matrix_get(z_coordinates,x,y);
	round_x = round((real)sp_3matrix_x(model)/2.0 - 0.5 + new_x);
	round_y = round((real)sp_3matrix_y(model)/2.0 - 0.5 + new_y);
	round_z = round((real)sp_3matrix_z(model)/2.0 - 0.5 + new_z);
	if (round_x >= 0 && round_x < sp_3matrix_x(model) &&
	    round_y >= 0 && round_y < sp_3matrix_y(model) &&
	    round_z >= 0 && round_z < sp_3matrix_z(model)) {
	  sp_3matrix_set(model,round_x,round_y,round_z,
			 sp_3matrix_get(model,round_x,round_y,round_z)+w*sp_matrix_get(slice,x,y));
	  sp_3matrix_set(weight,round_x,round_y,round_z,sp_3matrix_get(weight,round_x,round_y,round_z)+w);
	}
      }//endif
    }
  }
}

void test_blur() {
  int i_device = cuda_get_device();
  printf("device id = %d\n", i_device);
  /* test blur */
  const int image_side = 10;
  const int N_3d = pow(image_side, 3);

  real *image = malloc(N_3d*sizeof(real));
  for (int i = 0; i < N_3d; i++) {
    image[i] = 0.;
  }
  //image[image_side*image_side*image_side/2 + image_side*image_side/2 + image_side/2] = 1.;
  image[image_side*image_side*2 + image_side*3 + 3] = 1.;
  image[image_side*image_side*7 + image_side*8 + 5] = 1.;
  image[image_side*image_side*5 + image_side*5 + 5] = -1.;
  FILE *blur_out = fopen("debug/blur_before.data", "wp");
  for (int i = 0; i < N_3d; i++) {
    fprintf(blur_out, "%g\n", image[i]);
  }
  fclose(blur_out);
  real *d_image;
  cuda_allocate_real(&d_image, N_3d);
  cuda_copy_real_to_device(image, d_image, N_3d);


  int *mask = malloc(N_3d*sizeof(int));
  for (int i = 0; i < N_3d; i++) {
    mask[i] = 1;
  }
  int *d_mask;
  cuda_allocate_int(&d_mask, N_3d);
  cuda_copy_int_to_device(mask, d_mask, N_3d);

  cuda_blur_model(d_image, image_side, 1.);

  cuda_copy_real_to_host(image, d_image, N_3d);
  blur_out = fopen("debug/blur_after.data", "wp");
  for (int i = 0; i < N_3d; i++) {
    fprintf(blur_out, "%g\n", image[i]);
  }
  fclose(blur_out);
  exit(0);
  /* done testing blur */
}

void test_weight_map() {
  int image_side = 100;
  real width = 20.;
  real falloff = 10.;

  real *d_weight_map;
  cuda_allocate_weight_map(&d_weight_map, image_side);

  cuda_calculate_weight_map_ring(d_weight_map, image_side, 0., 0., width, falloff);

  real *weight_map = malloc(image_side*image_side*sizeof(real));
  cuda_copy_real_to_host(weight_map, d_weight_map, image_side*image_side);

  FILE *weight_map_out = fopen("debug/weight_map.data", "wp");
  for (int i = 0; i < image_side*image_side; i++) {
    fprintf(weight_map_out, "%g\n", weight_map[i]);
  }
  fclose(weight_map_out);
  exit(0);
}

Configuration read_configuration_file(const char *filename)
{
  Configuration config_out;
  config_t config;
  config_init(&config);
  if (!config_read_file(&config,filename)) {
    fprintf(stderr,"%d - %s\n",
	   config_error_line(&config),
	   config_error_text(&config));
    config_destroy(&config);
    exit(1);
  }
  config_lookup_int(&config,"model_side",&config_out.model_side);
  config_lookup_int(&config,"read_stride",&config_out.read_stride);
  config_lookup_float(&config,"wavelength",&config_out.wavelength);
  config_lookup_float(&config,"pixel_size",&config_out.pixel_size);
  config_lookup_int(&config,"detector_size",&config_out.detector_size);
  config_lookup_float(&config,"detector_distance",&config_out.detector_distance);
  config_lookup_int(&config,"rotations_n",&config_out.rotations_n);
  config_lookup_float(&config,"sigma_start",&config_out.sigma_start);
  config_lookup_float(&config,"sigma_final",&config_out.sigma_final);
  config_lookup_int(&config,"sigma_half_life",&config_out.sigma_half_life);
  config_lookup_int(&config,"slice_chunk",&config_out.slice_chunk);
  config_lookup_int(&config,"N_images",&config_out.N_images);
  config_lookup_int(&config,"max_iterations",&config_out.max_iterations);
  config_lookup_bool(&config,"blur_image",&config_out.blur_image);
  config_lookup_float(&config,"blur_sigma",&config_out.blur_sigma);
  config_lookup_string(&config,"mask_file",&config_out.mask_file);
  config_lookup_string(&config,"image_prefix",&config_out.image_prefix);
  config_lookup_bool(&config,"normalize_images",&config_out.normalize_images);
  config_lookup_bool(&config,"known_intensity",&config_out.known_intensity);
  config_lookup_int(&config,"model_input",&config_out.model_input);
  config_lookup_float(&config,"initial_model_noise",&config_out.initial_model_noise);
  config_lookup_string(&config,"model_file",&config_out.model_file);
  config_lookup_string(&config, "init_rotations", &config_out.init_rotations_file);
  config_lookup_bool(&config,"exclude_images",&config_out.exclude_images);
  config_lookup_float(&config,"exclude_ratio",&config_out.exclude_ratio);
  const char *diff_type_string = malloc(20*sizeof(char));
  config_lookup_string(&config,"diff_type",&diff_type_string);
  if (strcmp(diff_type_string, "absolute") == 0) {
    config_out.diff = absolute;
  } else if (strcmp(diff_type_string, "poisson") == 0) {
    config_out.diff = poisson;
  } else if (strcmp(diff_type_string, "relative") == 0) {
    config_out.diff = relative;
  }
  config_lookup_float(&config,"model_blur",&config_out.model_blur);

  config_out.pixel_size /= config_out.read_stride;
  return config_out;
}

sp_matrix **read_images(Configuration conf, sp_imatrix **masks)
{
  sp_matrix **images = malloc(conf.N_images*sizeof(sp_matrix *));
  //masks = malloc(conf.N_images*sizeof(sp_imatrix *));
  Image *img;
  real *intensities = malloc(conf.N_images*sizeof(real));
  char buffer[1000];

  for (int i = 0; i < conf.N_images; i++) {
    intensities[i] = 1.0;
  }

  for (int i = 0; i < conf.N_images; i++) {
    sprintf(buffer,"%s%.4d.h5", conf.image_prefix, i);
    img = sp_image_read(buffer,0);

    /* blur image if enabled */
    if (conf.blur_image == 1) {
      Image *tmp = sp_gaussian_blur(img,conf.blur_sigma);
      sp_image_free(img);
      img = tmp;
    }

    images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
    masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);

    for (int pixel_i = 0; pixel_i < sp_image_size(img); pixel_i++) {
      if (sp_real(img->image->data[pixel_i]) < 0.) {
	sp_real(img->image->data[pixel_i]) = 0.;
      }
    }

    /*
    pixel_sum += sp_cabs(sp_image_get(img,
				      (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,
				      (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0));
    */

    real pixel_sum;
    int mask_sum;
    for (int x = 0; x < conf.model_side; x++) {
      for (int y = 0; y < conf.model_side; y++) {
	pixel_sum = 0.0;
	mask_sum = 0;
	for (int xb = 0; xb < conf.read_stride; xb++) {
	  for (int yb = 0; yb < conf.read_stride; yb++) {
	    pixel_sum += sp_cabs(sp_image_get(img, sp_image_x(img)/2 - conf.model_side*conf.read_stride/2 + x*conf.read_stride + xb,
					     sp_image_y(img)/2 - conf.model_side*conf.read_stride/2 + y*conf.read_stride + yb, 0));
	    mask_sum += sp_image_mask_get(img, sp_image_x(img)/2 - conf.model_side*conf.read_stride/2 + x*conf.read_stride + xb,
					 sp_image_y(img)/2 - conf.model_side*conf.read_stride/2 + y*conf.read_stride + yb, 0);
	    /*
	    pixel_sum += sp_cabs(sp_image_get(img,(int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,(int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0));
	    mask_sum += sp_image_mask_get(img,(int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,(int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0);
	    */
	  }
	}
	if (mask_sum > 1) {
	  sp_matrix_set(images[i],x,y,pixel_sum/(real)mask_sum);
	  sp_imatrix_set(masks[i],x,y,1);
	} else {
	  sp_matrix_set(images[i],x,y,0.);
	  sp_imatrix_set(masks[i],x,y,0);
	}
      }
    }
    sp_image_free(img);
  }
  return images;
}

/* init mask */
sp_imatrix *read_mask(Configuration conf)
{
  sp_imatrix *mask = sp_imatrix_alloc(conf.model_side,conf.model_side);;
  Image *mask_in = sp_image_read(conf.mask_file,0);
  /* read and rescale mask */
  int mask_sum;
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      mask_sum = 0;
      for (int xb = 0; xb < conf.read_stride; xb++) {
	for (int yb = 0; yb < conf.read_stride; yb++) {
	  if (sp_cabs(sp_image_get(mask_in, sp_image_x(mask_in)/2 - conf.model_side*conf.read_stride/2 + x*conf.read_stride + xb,
				   sp_image_y(mask_in)/2 - conf.model_side*conf.read_stride/2 + y*conf.read_stride + yb, 0))) {
	    mask_sum += 1;
	  }
	}
      }
      if (mask_sum > 1) {
	sp_imatrix_set(mask,x,y,1);
      } else {
	sp_imatrix_set(mask,x,y,0);
      }
    }
  }
  /*
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      if (sp_cabs(sp_image_get(mask_in,
			       (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+
				     sp_image_x(mask_in)/2-0.5),
			       (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+
				     sp_image_y(mask_in)/2-0.5),0)) == 0.0) {
	sp_imatrix_set(mask,x,y,0);
      } else {
	sp_imatrix_set(mask,x,y,1);
      }
    }
  }
  */
  sp_image_free(mask_in);
  
  /* mask out everything outside the central sphere */
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      if (sqrt(pow((real)x - (real)conf.model_side/2.0+0.5,2) +
	       pow((real)y - (real)conf.model_side/2.0+0.5,2)) >
	  conf.model_side/2.0) {
	sp_imatrix_set(mask,x,y,0);
      }
    }
  }
  return mask;
}

/* normalize images so average pixel value is 1.0 */
void normalize_images(sp_matrix **images, sp_imatrix *mask, Configuration conf)
{
  real sum, count;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    sum = 0.;
    count = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] == 1) {
	sum += images[i_image]->data[i];
	count += 1.;
      }
    }
    sum = count / sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}

void normalize_images_central_part(sp_matrix **images, sp_imatrix *mask, real radius, Configuration conf) {
  const int x_max = conf.model_side;
  const int y_max = conf.model_side;
  sp_imatrix * central_mask = sp_imatrix_alloc(x_max, y_max);
  real r;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      r = pow(x-x_max/2+0.5, 2) + pow(y-y_max/2+0.5, 2);
      if (r < pow(radius,2)) {
	sp_imatrix_set(central_mask, x, y, 1);
      } else {
	sp_imatrix_set(central_mask, x, y, 1);
      }
    }
  }
  real sum, count;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    sum = 0.;
    count = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] == 1 && central_mask->data[i] == 1) {
	sum += images[i_image]->data[i];
	count += 1.;
      }
    }
    sum = (real) count / sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}

/* normalize images so average pixel value is 1.0 */
void normalize_images_individual_mask(sp_matrix **images, sp_imatrix **masks,
				      Configuration conf)
{
  real sum, count;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    sum = 0.;
    count = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 1) {
	sum += images[i_image]->data[i];
	count += 1.;
      }
    }
    sum = (real)count / sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}

/* this normalization is intended for use when known_intensities is set */
void normalize_images_preserve_scaling(sp_matrix ** images, sp_imatrix *mask, Configuration conf) {
  int N_2d = conf.model_side*conf.model_side;
  real sum = 0.;
  real count = 0.;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] == 1) {
	sum += images[i_image]->data[i];
	count += 1.;
      }
    }
  }
  sum = (count*(real)conf.N_images) / sum;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}

hid_t open_state_file(char *filename) {
  hid_t file_id, space_id, dataset_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  //init iteration
  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/iteration", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  int iteration_start_value = -1;
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration_start_value);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fflush(file_id, H5F_SCOPE_GLOBAL);

  return file_id;
}

void write_state_file_iteration(hid_t file_id, int value) {
  hid_t dataset_id;

  hsize_t file_size;
  H5Fget_filesize(file_id, &file_size);
  
  dataset_id = H5Dopen(file_id, "/iteration", H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  H5Dclose(dataset_id);
  H5Fflush(file_id, H5F_SCOPE_GLOBAL);
}

void close_state_file(hid_t file_id) {
  H5Fclose(file_id);
}

int main(int argc, char **argv)
{
  /*
  cuda_choose_best_device();
  cuda_print_device_info();
  cuda_test_interpolate_set();
  exit(0);
  */

  signal(SIGINT, nice_exit);
  //cuda_set_device(1);
  //cuda_set_device(cuda_get_best_device());
  cuda_choose_best_device();
  cuda_print_device_info();

  struct stat sb;
  if (stat("output", &sb) != 0) {
    if (mkdir("output",0777) == 0) {
      printf("Created directory: output\n");
    } else {
      printf("Failed to create directory: output\n");
    }
  }
  if (stat("debug", &sb) != 0) {
    if (mkdir("debug",0777) == 0) {
      printf("Created directory: output\n");
    } else {
      printf("Failed to create directory: output\n");
    }
  }

  //test_blur();
  //test_weight_map();

  hid_t state_file = open_state_file("output/state.h5");
  write_state_file_iteration(state_file, 2);
  write_state_file_iteration(state_file, 4);

  //signal(SIGKILL, nice_exit);
  Configuration conf;
  if (argc > 1) {
    conf = read_configuration_file(argv[1]);
  } else {
    conf = read_configuration_file("emc.conf");
  }
  const int start_iteration = 0;
  const int N_images = conf.N_images;
  const int slice_chunk = conf.slice_chunk;
  const int output_period = 10;
  const int n = conf.rotations_n;
  const int N_2d = conf.model_side*conf.model_side;
  const int N_3d = conf.model_side*conf.model_side*conf.model_side;
  char buffer[1000];

  Quaternion **rotations;
  real *weights;
  const int N_slices = generate_rotation_list(n,&rotations,&weights);
  real *d_weights;
  
  cuda_allocate_real(&d_weights, N_slices);
  cuda_copy_real_to_device(weights, d_weights, N_slices);
  printf("%d rotations sampled\n",N_slices);
  /* outpus weigths */
  /* now h5
  FILE *weights_file = fopen("debug/weights.data", "wp");
  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
    fprintf(weights_file, "%g\n", weights[i_slice]);
  }
  fclose(weights_file);
  */
  write_1d_array_hdf5("debug/weights.h5", weights, N_slices);

  /* output rotations */
  FILE *rotations_file = fopen("output/rotations.data", "wp");
  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
    fprintf(rotations_file, "%g %g %g %g\n", rotations[i_slice]->q[0],rotations[i_slice]->q[1], rotations[i_slice]->q[2], rotations[i_slice]->q[3]);
  }
  fclose(rotations_file);

  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  //  gsl_rng_set(rng,time(NULL));
  // Reproducible "random" numbers
  gsl_rng_set(rng,0);

  /* read images */
  sp_imatrix **masks = malloc(conf.N_images*sizeof(sp_imatrix *));
  sp_matrix **images = read_images(conf,masks);
  sp_imatrix * mask = read_mask(conf);

  /* try extending the mask from the center */
  /*
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      real r = (int)sqrt(pow((real)x - conf.model_side/2.0 + 0.5,2) +
			 pow((real)y - conf.model_side/2.0 + 0.5,2));
      if (r < 64.0) {
	sp_imatrix_set(mask,x,y,0);
      }
    }
  }
  */
  /* Create output directories in case they don't exist */
  if (conf.normalize_images) {
    if (conf.known_intensity) {
      normalize_images_preserve_scaling(images, mask, conf);
    } else {
      //normalize_images(images, mask, conf);
      //normalize_images_individual_mask(images, masks, conf);
      normalize_images_central_part(images, mask, 20., conf);
    }
  }
  /* output images after preprocessing */

  real image_max = 0.;
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] == 1 && images[i_image]->data[i] > image_max) {
	image_max = images[i_image]->data[i];
      }
    }
  }

  printf("image_max = %g\n", image_max);
  Image *write_image = sp_image_alloc(conf.model_side,conf.model_side,1);
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      //if (masks[i_image]->data[i]) {
      if (mask->data[i]) {
	sp_real(write_image->image->data[i]) = images[i_image]->data[i];
      } else {
	sp_real(write_image->image->data[i]) = 0.0;
      }
      //write_image->mask->data[i] = masks[i_image]->data[i];
      write_image->mask->data[i] = mask->data[i];
    }
    write_image->image->data[0] = sp_cinit(image_max, 0.);
    sprintf(buffer, "debug/image_%.4d.png", i_image);
    sp_image_write(write_image, buffer, SpColormapJet|SpColormapLogScale);
    sprintf(buffer, "debug/image_%.4d.h5", i_image);
    sp_image_write(write_image, buffer, 0);
  }
  sp_image_free(write_image);


  sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  calculate_coordinates(conf.model_side, conf.pixel_size*conf.read_stride, conf.detector_distance, conf.wavelength,
			x_coordinates, y_coordinates, z_coordinates);


  /* calculate correlation stuff */
  /*
  sp_matrix *corr_average = sp_matrix_alloc(conf.model_side, conf.model_side);
  sp_matrix *corr_scale = sp_matrix_alloc(conf.model_side, conf.model_side);
  calculate_normalization(images, N_images, corr_average, corr_scale);
  */
  /* create and fill model */
  Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
  sp_3matrix *model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  /*
  real model_d = 1.0/(conf.pixel_size*(real)conf.detector_size/conf.detector_distance*
		      conf.wavelength);
  */
  sp_3matrix *weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  const int N_model = conf.model_side*conf.model_side*conf.model_side;

  //change later to random rotations



  for (int i = 0; i < N_model; i++) {
    model->data[i] = 0.0;
    weight->data[i] = 0.0;
  }

  if (conf.model_input == 0) {
    printf("uniform density model\n");
    for (int i = 0; i < N_model; i++) {
      //model->data[i] = 1.0;
      model->data[i] = gsl_rng_uniform(rng);
    }
  } else if (conf.model_input == 3) {
    printf("radial average model\n");
    real *radavg = malloc(conf.model_side/2*sizeof(real));
    int *radavg_count = malloc(conf.model_side/2*sizeof(int));
    int r;
    for (int i = 0; i < conf.model_side/2; i++) {
      radavg[i] = 0.0;
      radavg_count[i] = 0;
    }
    for (int i_image = 0; i_image < N_images; i_image++) {
      for (int x = 0; x < conf.model_side; x++) {
	for (int y = 0; y < conf.model_side; y++) {
	  if (sp_matrix_get(images[i_image],x,y) >= 0.) {
	    if (isinf(sp_matrix_get(images[i_image],x,y))) {
	      printf("%d[%d,%d] is inf\n", i_image, x, y);
	    }
	    if (isnan(sp_matrix_get(images[i_image],x,y))) {
	      printf("%d[%d,%d] is nan\n", i_image, x, y);
	    }
	    r = (int)sqrt(pow((real)x - conf.model_side/2.0 + 0.5,2) +
			  pow((real)y - conf.model_side/2.0 + 0.5,2));
	    if (r < conf.model_side/2.0) {
	      radavg[r] += sp_matrix_get(images[i_image],x,y);
	      radavg_count[r] += 1;
	    }
	  }
	}
      }
    }
    for (int i = 0; i < conf.model_side/2; i++) {
      if (radavg_count[i] > 0) {
	radavg[i] /= (real) radavg_count[i];
      } else {
	radavg[i] = 0.0;
      }
      //printf("%d: %g\n", i, radavg[i]);
    }
    real rad;
    for (int x = 0; x < conf.model_side; x++) {
      for (int y = 0; y < conf.model_side; y++) {
	for (int z = 0; z < conf.model_side; z++) {
	  rad = sqrt(pow((real)x - conf.model_side/2.0 + 0.5,2) +
		     pow((real)y - conf.model_side/2.0 + 0.5,2) +
		     pow((real)z - conf.model_side/2.0 + 0.5,2));
	  r = (int)rad;
	  if (r < conf.model_side/2.0) {
	    sp_3matrix_set(model,x,y,z,(radavg[r]*(1.0 - (rad - (real)r)) +
					radavg[r+1]*(rad - (real)r)) * (1. + conf.initial_model_noise*gsl_rng_uniform(rng)));
	  } else {
	    sp_3matrix_set(model,x,y,z,-1.0);
	  }
	}
      }
    }
  } else if (conf.model_input == 1) {
    printf("random orientations model\n");
    Quaternion *random_rot;
    for (int i = 0; i < N_images; i++) {
      random_rot = quaternion_random(rng);
      insert_slice(model, weight, images[i], mask, 1.0, random_rot,
		   x_coordinates, y_coordinates, z_coordinates);
      free(random_rot);
    }
    for (int i = 0; i < N_model; i++) {
      if (weight->data[i] > 0.0) {
	model->data[i] /= (weight->data[i]);
      } else {
	model->data[i] = 0.0;
      }
    }
  } else if (conf.model_input == 2) {
    printf("model from file %s\n",conf.model_file);
    Image *model_in = sp_image_read(conf.model_file,0);
    if (conf.model_side != sp_image_x(model_in) ||
	conf.model_side != sp_image_y(model_in) ||
	conf.model_side != sp_image_z(model_in)) {
      printf("Input model is of wrong size.\n");
      exit(1);
    }
    for (int i = 0; i < N_model; i++) {
      model->data[i] = sp_cabs(model_in->image->data[i]);
    }
    sp_image_free(model_in);
  } else if (conf.model_input == 4) {
    printf("given orientations model\n");
    FILE *given_rotations_file = fopen(conf.init_rotations_file, "r");
    Quaternion *this_rotation = quaternion_alloc();
    for (int i_image = 0; i_image < N_images; i_image++) {
      fscanf(given_rotations_file, "%g %g %g %g\n", &(this_rotation->q[0]), &(this_rotation->q[1]), &(this_rotation->q[2]), &(this_rotation->q[3]));
      //printf("(%d) : %g, %g, %g, %g\n", i_image, this_rotation->q[0], this_rotation->q[1], this_rotation->q[2], this_rotation->q[3]);
      insert_slice(model, weight, images[i_image], mask, 1., this_rotation,
		   x_coordinates, y_coordinates, z_coordinates);
    }
    free(this_rotation);
    fclose(given_rotations_file);
    
    for (int i = 0; i < N_model; i++) {
      if (weight->data[i] > 0.) {
	model->data[i] /= weight->data[i];
      }else {
	model->data[i] = -1.;
      }
    }
  }

  real *scaling = malloc(N_images*N_slices*sizeof(real));
  for (int i = 0; i < N_images; i++) {
    scaling[i] = 1.0;
  }

  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(model->data[i],0.0);
    if (weight->data[i] > 0.0) {
      model_out->mask->data[i] = 1;
    } else {
      model_out->mask->data[i] = 0;
    }
  }
  sprintf(buffer,"output/model_init.h5");
  sp_image_write(model_out,buffer,0);
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(weight->data[i],0.0);
  }
  sprintf(buffer,"output/model_init_weight.h5");
  sp_image_write(model_out,buffer,0);
  printf("wrote initial model\n");

  sp_matrix *radius = sp_matrix_alloc(conf.model_side,conf.model_side);
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
	sp_matrix_set(radius,x,y,sqrt(pow((real) x - conf.model_side/2.0 + 0.5, 2) +
				      pow((real) y - conf.model_side/2.0 + 0.5, 2)));
    }
  }
  
  /*real respons[N_slices][N_images];*/
  real *respons = malloc(N_slices*N_images*sizeof(real));
  real sum, total_respons;
  FILE *likelihood = fopen("likelihood.data","wp");

  real * slices;
  cuda_allocate_slices(&slices,conf.model_side,slice_chunk); //was N_slices before
  //real * slices_on_host = malloc(N_slices*N_2d*sizeof(real));
  real * d_model;
  cuda_allocate_model(&d_model,model);
  cuda_normalize_model(model, d_model);
  real * d_model_updated;
  real * d_model_tmp;
  cuda_allocate_model(&d_model_updated,model);
  real * d_weight;
  cuda_allocate_model(&d_weight,weight);
  int * d_mask;
  cuda_allocate_mask(&d_mask,mask);
  real * d_rotations;
  cuda_allocate_rotations(&d_rotations,rotations,N_slices);
  real * d_x_coord;
  real * d_y_coord;
  real * d_z_coord;
  cuda_allocate_coords(&d_x_coord,
		       &d_y_coord,
		       &d_z_coord,
		       x_coordinates,
		       y_coordinates, 
		       z_coordinates);
  real * d_images;
  cuda_allocate_images(&d_images,images,N_images);
  int * d_masks;
  cuda_allocate_masks(&d_masks,masks,N_images);
  real * d_respons;
  cuda_allocate_real(&d_respons,N_slices*N_images);
  //real * d_scaling;
  //cuda_allocate_scaling(&d_scaling,N_images);
  real * d_scaling;
  cuda_allocate_scaling_full(&d_scaling, N_images, N_slices);
  real *d_weighted_power;
  cuda_allocate_real(&d_weighted_power,N_images);
  real *fit = malloc(N_images*sizeof(real));
  real *d_fit;
  cuda_allocate_real(&d_fit,N_images);
  real *fit_best_rot = malloc(N_images*sizeof(real));
  real *d_fit_best_rot;
  cuda_allocate_real(&d_fit_best_rot, N_images);
  int *active_images = malloc(N_images*sizeof(int));
  int *d_active_images;
  cuda_allocate_int(&d_active_images,N_images);
  real *d_radius;
  cuda_allocate_real(&d_radius, N_2d);
  cuda_copy_real_to_device(radius->data, d_radius, N_2d);
  real *radial_fit = malloc(conf.model_side/2*sizeof(real));
  real *radial_fit_weight = malloc(conf.model_side/2*sizeof(real));
  real *d_radial_fit;
  real *d_radial_fit_weight;
  cuda_allocate_real(&d_radial_fit, conf.model_side/2);
  cuda_allocate_real(&d_radial_fit_weight, conf.model_side/2);

  int *best_rotation = malloc(N_images*sizeof(int));
  int *d_best_rotation;
  cuda_allocate_int(&d_best_rotation, N_images);
  
  real *sorted_resp = malloc(N_slices*sizeof(real));
  real *total_sorted_resp = malloc(N_slices*sizeof(real));
  real *average_resp = malloc(N_slices*sizeof(real));

  FILE *best_rot_file = fopen("output/best_rot.data", "wp");
  FILE *fit_file = fopen("output/fit.data","wp");
  FILE *fit_best_rot_file = fopen("output/fit_best_rot.data","wp");
  FILE *radial_fit_file = fopen("output/radial_fit.data","wp");
  FILE *sorted_resp_file = fopen("output/total_resp.data","wp");
  FILE *average_resp_file;// = fopen("output/average_resp.data","wp");

  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 0) {
	images[i_image]->data[i] = -1.0;
      }
    }
  }

  real * d_masked_images;
  cuda_allocate_images(&d_masked_images,images,N_images);

  int current_chunk;

  
  /*
  real sigma_half_life = 25; // 1000.
  real sigma_start = 0.12;   // relative: 0.09 -> 0.05
  real sigma_final = 0.0; // 0.2
  enum diff_type diff = poisson;
  */
  /*
  real sigma_half_life = 25; // 1000.
  real sigma_start = 0.12;   // relative: 0.09 -> 0.05
  real sigma_final = 0.; // 0.2
  enum diff_type diff = poisson;
  */
  real *d_weight_map;
  cuda_allocate_weight_map(&d_weight_map, conf.model_side);
  real weight_map_radius, weight_map_falloff;
  real weight_map_radius_start = 20.;
  real weight_map_radius_final = 20.;

  real sigma;
  for (int iteration = start_iteration; iteration < conf.max_iterations; iteration++) {
    if (quit_requested == 1) {
      break;
    }
    printf("\niteration %d\n", iteration);
    write_state_file_iteration(state_file, iteration);

    //conf.sigma = sigma_final + (sigma_start-sigma_final)*exp(-iteration/sigma_half_life*log(2.));
    sigma = conf.sigma_final + (conf.sigma_start-conf.sigma_final)*exp(-iteration/(float)conf.sigma_half_life*log(2.));
    printf("sigma = %g\n", sigma);

    //weight_map_radius = 20.;
    weight_map_radius = weight_map_radius_start + ((weight_map_radius_final-weight_map_radius_start) *
						   ((real)iteration / ((real)conf.sigma_half_life)));
    weight_map_falloff = 0.;
    //cuda_calculate_weight_map_ring(d_weight_map, conf.model_side, 0., 0., weight_map_radius, weight_map_falloff);
    cuda_calculate_weight_map_ring(d_weight_map, conf.model_side, 8., 0., 40., 0.);
    printf("weight_map: radius = %g, falloff = %g\n", weight_map_radius, weight_map_falloff);

    //sum = cuda_model_max(d_model, N_model);
    //printf("model max = %g\n",sum);
    
    //sum = cuda_model_sum(d_model, N_model);
    //sum = cuda_model_average(d_model, N_3d);
    //printf("model_average = %f\n", sum);

    /* start calculate many fits */
    int radial_fit_n = 1;
    cuda_set_to_zero(d_fit,N_images);
    cuda_set_to_zero(d_radial_fit,conf.model_side/2);
    cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2);
    cuda_calculate_best_rotation(d_respons, d_best_rotation, N_images, N_slices);
    cuda_copy_int_to_host(best_rotation, d_best_rotation, N_images);
    for (int i_image = 0; i_image < N_images; i_image++) {
      fprintf(best_rot_file, "%d ", best_rotation[i_image]);
    }
    fprintf(best_rot_file, "\n");
    fflush(best_rot_file);

    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      cuda_get_slices(model,d_model,slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);

      /* start output best fitting slices */

      Image *slice_out_img = sp_image_alloc(conf.model_side, conf.model_side, 1);
      real *slice_out = malloc(N_2d*sizeof(real));

      for (int i_image = 0; i_image < N_images; i_image++) {
	if (best_rotation[i_image] >= slice_start && best_rotation[i_image] < slice_start + current_chunk) {
	  //printf("best_rotation[%d] = %d (%d) -> %d\n", i_image, best_rotation[i_image], N_slices, best_rotation[i_image] - slice_start);

	  cuda_copy_real_to_host(slice_out, &slices[(best_rotation[i_image] - slice_start)*N_2d], N_2d);

	  for (int i = 0; i < N_2d; i++) {
	    slice_out_img->image->data[i] = sp_cinit(slice_out[i], 0.);
	  }
	  sprintf(buffer, "output/best_slice_%.4d_%.4d.h5", iteration, i_image);
	  sp_image_write(slice_out_img, buffer, 0);

	}
      }

      sp_image_free(slice_out_img);
      free(slice_out);

      /* end output best fitting slices */

      
      cuda_calculate_fit(slices, d_images, d_mask, d_scaling,
			 d_respons, d_fit, sigma, N_2d, N_images,
			 slice_start, current_chunk);

      cuda_calculate_fit_best_rot(slices, d_images, d_mask, d_scaling,
				  d_best_rotation, d_fit_best_rot, N_2d, N_images,
				  slice_start, current_chunk);
      
      //if (iteration % radial_fit_n == 0 && iteration != 0 || iteration == conf.max_iterations-1) {
      if (iteration % radial_fit_n == 0 && iteration != 0) {
	cuda_calculate_radial_fit(slices, d_images, d_mask,
				  d_scaling, d_respons, d_radial_fit,
				  d_radial_fit_weight, d_radius,
				  N_2d, conf.model_side, N_images, slice_start,
				  current_chunk);
      }
    }

    /* test scaling by outputting two */
    /*
    real *out = malloc(N_2d*sizeof(real));
    cuda_copy_real_to_host(out, d_images, N_2d);
    sprintf(buffer, "debug/image_0_%.4d.data", iteration);
    FILE *fi = fopen(buffer, "wp");
    for (int i1 = 0; i1 < conf.model_side; i1++) {
      for (int i2 = 0; i2 < conf.model_side; i2++) {
	fprintf(fi, "%g ", out[i1*conf.model_side + i2]);
      }
      fprintf(fi, "\n");
    }
    fclose(fi);
    cuda_copy_real_to_host(out, slices, N_2d);
    sprintf(buffer, "debug/slice_0_%.4d.data", iteration);
    fi = fopen(buffer, "wp");
    for (int i1 = 0; i1 < conf.model_side; i1++) {
      for (int i2 = 0; i2 < conf.model_side; i2++) {
	fprintf(fi, "%g ", out[i1*conf.model_side + i2]);
      }
      fprintf(fi, "\n");
    }
    fclose(fi);
    free(out);
    */

    /* output all slices at choosen iteration */
    if (iteration == 1000) {
      real *h_slices = malloc(slice_chunk*N_2d*sizeof(real));
      Image *out = sp_image_alloc(conf.model_side, conf.model_side, 1);	
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	cuda_get_slices(model, d_model, slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	
	cuda_copy_real_to_host(h_slices, slices, current_chunk*N_2d);

	for (int i_slice = 0; i_slice < current_chunk; i_slice++) {
	  if (i_slice % 1000 == 0) {
	    printf("output %d %d\n", slice_start/slice_chunk, i_slice);
	  }
	  for (int i = 0; i < N_2d; i++) {
	    out->image->data[i] = sp_cinit(h_slices[i_slice*N_2d + i], 0.);
	  }
	  sprintf(buffer, "debug/slice_%.7d.h5", slice_start + i_slice);
	  sp_image_write(out, buffer, 0);
	}
      }
      exit(0);
    }
    
      

    cuda_copy_real_to_host(fit, d_fit, N_images);
    cuda_copy_real_to_host(fit_best_rot, d_fit_best_rot, N_images);
    for (int i_image = 0; i_image < N_images; i_image++) {
      fprintf(fit_file, "%g ", fit[i_image]);
      fprintf(fit_best_rot_file, "%g ", fit_best_rot[i_image]);
    }
    fprintf(fit_file, "\n");
    fprintf(fit_best_rot_file, "\n");
    fflush(fit_file);
    fflush(fit_best_rot_file);

    /* finish calculating radial fit */
    //if ((iteration % radial_fit_n == 0 && iteration != 0) || iteration == conf.max_iterations-1) {
    if ((iteration % radial_fit_n == 0 && iteration != 0)) {
      cuda_copy_real_to_host(radial_fit, d_radial_fit, conf.model_side/2);
      cuda_copy_real_to_host(radial_fit_weight, d_radial_fit_weight, conf.model_side/2);
      for (int i = 0; i < conf.model_side/2; i++) {
	//printf("%d: %g %g\n", i, radial_fit[i], radial_fit_weight[i]);
	if (radial_fit_weight[i] > 0.0) {
	  radial_fit[i] /= radial_fit_weight[i];
	} else {
	  radial_fit[i] = 0.0;
	}
      }
      for (int i = 0; i < conf.model_side/2; i++) {
	fprintf(radial_fit_file, "%g ", radial_fit[i]);
      }
      fprintf(radial_fit_file, "\n");
      fflush(radial_fit_file);
    }
    /* end calculate many fits */

    /* start update scaling */
    //if (conf.known_intensity == 0 && 1 == 2) {
    if (conf.known_intensity == 0) {
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	cuda_get_slices(model, d_model, slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	
	cuda_update_scaling_full(d_images, slices, d_mask, d_scaling, d_weight_map, N_2d, N_images, slice_start, current_chunk, conf.diff);
      }

      /* output scaling */
      cuda_copy_real_to_host(scaling, d_scaling, N_images*N_slices);

      sprintf(buffer, "output/scaling_%.4d.h5", iteration);
      write_2d_array_hdf5(buffer, scaling, N_slices, N_images);
    }
    //fprintf(scaling_file, "\n");
    //fclose(scaling_file);
    /* end update scaling */

    /* start calculate responsabilities */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      
      cuda_get_slices(model,d_model,slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);

      cuda_calculate_responsabilities(slices, d_images, d_mask, d_weight_map,
				      sigma, d_scaling, d_respons, d_weights, 
				      N_2d, N_images, slice_start,
				      current_chunk, conf.diff);

    }
    cuda_calculate_responsabilities_sum(respons, d_respons, N_slices, N_images);
    cuda_normalize_responsabilities(d_respons, N_slices, N_images);
    cuda_copy_real_to_host(respons, d_respons, N_slices*N_images);

    /* output responsabilities */
    sprintf(buffer, "output/responsabilities_%.4d.h5", iteration);
    write_2d_array_trans_hdf5(buffer, respons, N_slices, N_images);

    /* output average responsabilities */
    sprintf(buffer, "output/average_resp_%.4d.data", iteration);
    average_resp_file = fopen(buffer, "wp");
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      average_resp[i_slice] = 0.;
    }
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      for (int i_image = 0; i_image < N_images; i_image++) {
	average_resp[i_slice] += respons[i_slice*N_images+i_image];
      }
    }
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      fprintf(average_resp_file, "%g\n", average_resp[i_slice]);
    }
    fclose(average_resp_file);
    

    /* output sorted responsabilities */
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      total_sorted_resp[i_slice] = 0.0;
    }
    for (int i_image = 0; i_image< N_images; i_image++) {
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	sorted_resp[i_slice] = respons[i_slice*N_images+i_image];
      }
      qsort(sorted_resp, N_slices, sizeof(real), compare_real);
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	total_sorted_resp[i_slice] += sorted_resp[i_slice];
      }
    }
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      fprintf(sorted_resp_file, "%g ", total_sorted_resp[i_slice] / (float) N_images);
    }
    fprintf(sorted_resp_file,"\n");
    fflush(sorted_resp_file);
    /* end calculate responsabilities */


    /* start calculate likelihood */
    total_respons = cuda_total_respons(d_respons,respons,N_images*N_slices);

    fprintf(likelihood,"%g\n",total_respons);
    //printf("likelihood = %g\n",total_respons);
    fflush(likelihood);
    /* end calculate likelihood */
  

    /* reset model */    
    cuda_reset_model(model,d_model_updated);
    cuda_reset_model(weight,d_weight);
    printf("models reset\n");


    /* start exclude images */
    if (iteration == 0) {
      for (int i_image = 0; i_image < N_images; i_image++) {
	active_images[i_image] = 1;
      }
    }
    if (conf.exclude_images == 1 && iteration > -1) {
      real *fit_copy = malloc(N_images*sizeof(real));
      memcpy(fit_copy,fit,N_images*sizeof(real));
      qsort(fit_copy, N_images, sizeof(real), compare_real);
      real threshold = fit_copy[(int)((real)N_images*conf.exclude_ratio)];
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (i_image != 0 && i_image % 10 == 0) printf(" ");
	if (fit[i_image] > threshold) {
	  active_images[i_image] = 1;
	  printf("1");
	} else {
	  active_images[i_image] = 0;
	  printf("0");
	}
      }
      printf("\n");
    }
    cuda_copy_int_to_device(active_images, d_active_images, N_images);
    /* end exclude images */

    /* start update scaling second time (test) */
    if (conf.known_intensity == 0) {
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	cuda_get_slices(model, d_model, slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	
	cuda_update_scaling_full(d_images, slices, d_mask, d_scaling, d_weight_map, N_2d, N_images, slice_start, current_chunk, conf.diff);
      }
    }
    /* end update scaling second time (test) */

    /* start update model */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      if ((slice_start/slice_chunk)%output_period == 0) {
	printf("update slices chunk %d\n", slice_start/slice_chunk);
      }

      cuda_get_slices(model, d_model, slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
		      slice_start, current_chunk);

      cuda_update_slices(d_images, slices, d_mask,
			 d_respons, d_scaling, d_active_images,
			 N_images, slice_start, current_chunk, N_2d,
			 model,d_model_updated, d_x_coord, d_y_coord,
			 d_z_coord, &d_rotations[slice_start*4],
			 d_weight,images);
    }
    d_model_tmp = d_model_updated;
    d_model_updated = d_model;
    d_model = d_model_tmp;

    cuda_divide_model_by_weight(model, d_model, d_weight);
    cuda_normalize_model(model, d_model);

    //sprintf(buffer, "debug/model_before_blur_%.4d.h5", iteration);
    //cuda_output_device_model(d_model, buffer, conf.model_side);
    //cuda_blur_model(d_model, conf.model_side, conf.model_blur);
    //sprintf(buffer, "debug/model_after_blur_%.4d.h5", iteration);
    //cuda_output_device_model(d_model, buffer, conf.model_side);

    cuda_copy_model(model, d_model);
    cuda_copy_model(weight, d_weight);

    /* write model */
    for (int i = 0; i < N_model; i++) {
      //model_out->image->data[i] = sp_cinit(model->data[i],0.0);
      if (weight->data[i] > 0.0 && model->data[i] > 0.) {
	model_out->mask->data[i] = 1;
	model_out->image->data[i] = sp_cinit(model->data[i],0.0);
      } else {
	model_out->mask->data[i] = 0;
	model_out->image->data[i] = sp_cinit(0., 0.);
      }
    }
    sprintf(buffer,"output/model_%.4d.h5", iteration);
    sp_image_write(model_out,buffer,0);
    /* write weight */
    for (int i = 0; i < N_model; i++) {
      model_out->image->data[i] = sp_cinit(weight->data[i], 0.);
      model_out->mask->data[i] = 1;
    }
    sprintf(buffer, "output/weight_%.4d.h5", iteration);
    sp_image_write(model_out, buffer, 0);

    /* end update model */
  }
  fclose(likelihood);
  fclose(best_rot_file);
  fclose(fit_file);
  fclose(fit_best_rot_file);
  //fclose(scaling_file);
  fclose(radial_fit_file);
  fclose(sorted_resp_file);
  //fclose(average_resp_file);

  cuda_reset_model(model,d_model_updated);
  cuda_reset_model(weight,d_weight);

  /*
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 0) {
	images[i_image]->data[i] = -1.0;
      }
    }
  }

  real * d_masked_images;
  cuda_allocate_images(&d_masked_images,images,N_images);
  */

  /* put together the model one last time for output.
     This time more of the middle data is used */

  //cuda_collapse_responsabilities(d_respons, N_slices, N_images);

  for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
    if (slice_start + slice_chunk >= N_slices) {
      current_chunk = N_slices - slice_start;
    } else {
      current_chunk = slice_chunk;
    }
    if ((slice_start/slice_chunk)%output_period == 0) {
      printf("update slices chunk %d\n", slice_start/slice_chunk);
    }
    /*
    cuda_get_slices(model, d_model, slices, d_rotations,
		    d_x_coord, d_y_coord, d_z_coord,
		    slice_start, current_chunk);
    */
    //cuda_copy_slice_chunk_to_device(slices_on_host, slices, slice_start, slice_chunk, N_2d);

    cuda_update_slices_final(d_masked_images, slices, d_mask,
			     d_respons, d_scaling, d_active_images,
			     N_images, slice_start, current_chunk, N_2d,
			     model,d_model_updated, d_x_coord, d_y_coord,
			     d_z_coord, &d_rotations[slice_start*4],
			     d_weight,images);

  }
  real *debug_model = malloc(conf.model_side*conf.model_side*conf.model_side*sizeof(real));
  real *debug_weight = malloc(conf.model_side*conf.model_side*conf.model_side*sizeof(real));
  cuda_copy_real_to_host(debug_model, d_model_updated, conf.model_side*conf.model_side*conf.model_side);
  cuda_copy_real_to_host(debug_weight, d_weight, conf.model_side*conf.model_side*conf.model_side);
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(debug_model[i],0.0);
  }
  sp_image_write(model_out,"debug/debug_model.h5",0);
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(debug_weight[i],0.0);
  }
  sp_image_write(model_out,"debug/debug_weight.h5",0);

  cuda_divide_model_by_weight(model, d_model_updated, d_weight);
  cuda_normalize_model(model, d_model_updated);  
  cuda_copy_model(model, d_model_updated);
  /* write output */
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(model->data[i],0.0);
    if (weight->data[i] > 0.0) {
      model_out->mask->data[i] = 1;
    } else {
      model_out->mask->data[i] = 0;
    }
  }
  model_out->scaled = 0;
  model_out->shifted = 0;
  model_out->phased = 0;
  model_out->detector->detector_distance = conf.detector_distance;
  model_out->detector->image_center[0] = conf.model_side/2. + 0.5;
  model_out->detector->image_center[1] = conf.model_side/2. + 0.5;
  model_out->detector->image_center[2] = conf.model_side/2. + 0.5;
  model_out->detector->pixel_size[0] = conf.pixel_size;
  model_out->detector->pixel_size[1] = conf.pixel_size;
  model_out->detector->pixel_size[2] = conf.pixel_size;
  model_out->detector->wavelength = conf.wavelength;
  
  sp_image_write(model_out,"output/model_final.h5",0);

  
  FILE *final_best_rotations_file = fopen("debug/final_best_rotations.data","wp");
  real highest_resp, this_resp;
  int final_best_rotation;
  for (int i_image = 0; i_image < N_images; i_image++) {
    final_best_rotation = 0;
    highest_resp = respons[0*N_images+i_image];
    for (int i_slice = 1; i_slice < N_slices; i_slice++) {
      this_resp = respons[i_slice*N_images+i_image];
      if (this_resp > highest_resp) {
	final_best_rotation = i_slice;
	highest_resp = this_resp;
      }
    }
    fprintf(final_best_rotations_file, "%g %g %g %g\n",
	    rotations[final_best_rotation]->q[0], rotations[final_best_rotation]->q[1],
	    rotations[final_best_rotation]->q[2], rotations[final_best_rotation]->q[3]);
  }
  fclose(final_best_rotations_file);
  close_state_file(state_file);
}
