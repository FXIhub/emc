#include <spimage.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include "emc.h"
#include "configuration.h"
#include <signal.h>
#include <sys/stat.h>
#include <hdf5.h>
#include <getopt.h>
#include <time.h>
#include <stdarg.h>

#define MAX_PATH_LENGTH 256
//const int MAX_PATH_LENGTH = 256;

static int quit_requested = 0;

/* Capture a crtl-c event to make a final iteration be
   run with the individual masked used in the compression.
   This is consistent with the final iteration when not
   interupted. Ctrl-c again will exit immediatley. */
void nice_exit(int sig) {
  if (quit_requested == 0) {
    quit_requested = 1;
  } else {
    exit(1);
  }
}

void error_exit_with_message(const char *message, ...) {
  va_list ap;
  va_start(ap, message);
  fprintf(stderr, "Error: ");
  vfprintf(stderr, message, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

void error_warning(const char *message, ...) {
  va_list ap;
  va_start(ap, message);
  fprintf(stderr, "Warning: ");
  vfprintf(stderr, message, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

/* Only used to provide to qsort. */
int compare_real(const void *pa, const void *pb) {
  real a = *(const real*)pa;
  real b = *(const real*)pb;

  if (a < b) {return -1;}
  else if (a > b) {return 1;}
  else {return 0;}
}

/* Writes any real type array in hdf5 format. */
void write_1d_real_array_hdf5(char *filename, real *array, int index1_max) {
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

/* Writes any int type array in hdf5 format. */
void write_1d_int_array_hdf5(char *filename, int *array, int index1_max) {
  hid_t file_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t space_id;
  hid_t dataset_id;
  hsize_t dim[1]; dim[0] = index1_max;
  
  space_id = H5Screate_simple(1, dim, NULL);
  dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}

/* Initialize a file that will contain the best scaling for each diffraction
   pattern for every iteratino. */
hid_t init_scaling_file(char *filename, int N_images, hid_t *file_id) {
  *file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {printf("Error creating file\n");}
  
  const int init_stack_size = 10;
  hsize_t dims[2]; dims[0] = init_stack_size; dims[1] = N_images;
  hsize_t maxdims[2]; maxdims[0] = H5S_UNLIMITED; maxdims[1] = N_images;

  hid_t dataspace = H5Screate_simple(2, dims, maxdims);
  if (dataspace < 0) error_exit_with_message("Error creating scaling dataspace\n");

  hid_t cparms = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(cparms, 2, dims);
  
  hid_t dataset = H5Dcreate(*file_id, "scaling", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, cparms, H5P_DEFAULT);
  if (dataset < 0) error_exit_with_message("Error creating scaling dataset\n");
  H5Pset_chunk_cache(H5Dget_access_plist(dataset), H5D_CHUNK_CACHE_NSLOTS_DEFAULT, N_images, 1);
  
  H5Sclose(dataspace);
  H5Pclose(cparms);
  return dataset;
}

/* Write the best scaling for each diffraction pattern to an already open
   hdf5 file. */
void write_scaling_to_file(const hid_t dataset, const int iteration, float *scaling, float *respons, int N_slices) {
  hid_t dataspace = H5Dget_space(dataset);
  if (dataspace < 0) error_exit_with_message("Can not create dataset when writing scaling.\n");
  hsize_t block[2];
  hsize_t mdims[2];
  H5Sget_simple_extent_dims(dataspace, block, mdims);
  
  /* Check if stack has enough space, otherwhise enlarge. */
  if (block[0] <= iteration) {
    while(block[0] <= iteration) {
      block[0] *= 2;
    }
    H5Dset_extent(dataset, block);
    H5Sclose(dataspace);
    dataspace = H5Dget_space(dataset);
    if (dataspace<0) error_exit_with_message("Error enlarging dataspace in scaling\n");
  }

  /* Now that we know the extent, find the best responsability and create an
     array with the scaling. */
  int N_images = block[1];
  float *data = malloc(N_images*sizeof(float));
  int best_index;
  float best_resp;
  for (int i_image = 0; i_image < N_images; i_image++) {
    best_resp = respons[i_image];
    best_index = 0;
    for (int i_slice = 1; i_slice < N_slices; i_slice++) {
      if (respons[i_slice*N_images + i_image] > best_resp) {
	best_resp = respons[i_slice*N_images + i_image];
	best_index = i_slice;
      }
    }
    data[i_image] = scaling[best_index*N_images + i_image];
  }

  /* Get the hdf5 hyperslab for the current iteration */
  block[0] = 1;
  hid_t memspace = H5Screate_simple(2, block, NULL);
  if (memspace < 0) error_exit_with_message("Error creating memspace when writing scaling\n");
  hsize_t offset[2] = {iteration, 0};
  hsize_t stride[2] = {1, 1};
  hsize_t count[2] = {1, 1};
  hid_t hyperslab = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, stride, count, block);
  if (hyperslab < 0) error_exit_with_message("Error selecting hyperslab in scaling\n");
  hid_t w = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data);
  if (w < 0) error_exit_with_message("Error writing scaling\n");
  free(data);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Fflush(dataset, H5F_SCOPE_GLOBAL);
}

/* Close scaling file. Call this after the last call to
   write_scaling_to_file. */
void close_scaling_file(hid_t dataset, hid_t file) {
  H5Sclose(dataset);
  H5Sclose(file);
}

/* Writes any real type 2D array in hdf5 format. */
void write_2d_real_array_hdf5(char *filename, real *array, int index1_max, int index2_max) {
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

/* Writes any real type 2D array in hdf5 format. Transposes the
   array before writing. */
void write_2d_real_array_hdf5_transpose(char *filename, real *array, int index1_max, int index2_max) {

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

/* Writes any real type 3D array in hdf5 format. */
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

/* Precalculate coordinates. These coordinates represent an
   Ewald sphere with the xy plane with liftoff in the z direction.
   These coordinates then only have to be rotated to get coordinates
   for expansion and compression. */
/* void calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength, */
/* 			   sp_matrix *x_coordinates, */
/* 			   sp_matrix *y_coordinates, sp_matrix *z_coordinates) { */
/*   const int x_max = side; */
/*   const int y_max = side; */
/*   real radius_in_pixels, radius_real, radius_fourier, radius_angle, z_liftoff_fourier; */
/*   real x_in_pixels, y_in_pixels, z_in_pixels; */
/*   //tabulate angle later */
/*   for (int x = 0; x < x_max; x++) { */
/*     for (int y = 0; y < y_max; y++) { */
/*       radius_in_pixels = sqrt(pow((real)(x-x_max/2)+0.5,2) + pow((real)(y-y_max/2)+0.5,2)); */
/*       radius_real = radius_in_pixels*pixel_size; */
/*       radius_angle = atan2(radius_real, detector_distance); */
/*       radius_fourier = sin(radius_angle)/wavelength; */
/*       z_liftoff_fourier = -(1. - cos(radius_angle))/wavelength; */
/*       //z_liftoff_fourier = (1. - cos(radius_angle))/wavelength; */

/*       x_in_pixels = (real)(x-x_max/2)+0.5; */
/*       y_in_pixels = (real)(y-y_max/2)+0.5; */
/*       z_in_pixels = z_liftoff_fourier/radius_fourier*radius_in_pixels; */
/*       sp_matrix_set(x_coordinates, x, y, x_in_pixels); */
/*       sp_matrix_set(y_coordinates, x, y, y_in_pixels); */
/*       sp_matrix_set(z_coordinates, x, y, z_in_pixels); */
/*       /\* sp_matrix_set(x_coordinates, x, y, y_in_pixels); *\/ */
/*       /\* sp_matrix_set(y_coordinates, x, y, x_in_pixels); *\/ */
/*       /\* sp_matrix_set(z_coordinates, x, y, z_in_pixels); *\/ */

/*     } */
/*   } */
/*   printf("liftoff: %f\n", sp_matrix_get(z_coordinates, 0, 0)); */
/* } */

void calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength,
			   sp_matrix *x_coordinates, sp_matrix *y_coordinates, sp_matrix *z_coordinates)
{
  const int x_max = side;
  const int y_max = side;

  real radius_at_edge = sqrt(2.)/wavelength * sqrt(1. - cos(atan2(pixel_size * (side/2. - 0.5), detector_distance)));

  real rescale_factor = side/2./radius_at_edge;
  
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      real x_pixels = x - x_max/2. + 0.5;
      real y_pixels = y - y_max/2. + 0.5;
      real x_meters = x_pixels * pixel_size;
      real y_meters = y_pixels * pixel_size;
      real radius_meters = sqrt(pow(x_meters, 2) + pow(y_meters, 2));
      real scattering_angle = atan2(radius_meters, detector_distance);
      real z_fourier = -1./wavelength * (1. - cos(scattering_angle));
      real radius_fourier = sqrt(pow(1./wavelength, 2) - pow(1./wavelength - fabs(z_fourier), 2));

      real x_fourier = x_meters * radius_fourier / radius_meters;
      real y_fourier = y_meters * radius_fourier / radius_meters;

      sp_matrix_set(x_coordinates, x, y, x_fourier*rescale_factor);
      sp_matrix_set(y_coordinates, x, y, y_fourier*rescale_factor);
      sp_matrix_set(z_coordinates, x, y, z_fourier*rescale_factor);
    }
  }
}

/* Like the compression step but only insert one slice into the model. This
   function is for exapmle used when initializing a model from random
   orientaitons. */
void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
		  sp_imatrix * mask, real w, Quaternion rot, sp_matrix *x_coordinates,
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
	/* This is a matrix multiplication of the x/y/z_coordinates with the
	   rotation matrix of the rotation rot. */
	new_x =
	  (rot.q[0]*rot.q[0] + rot.q[1]*rot.q[1] -
	   rot.q[2]*rot.q[2] - rot.q[3]*rot.q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (2.0*rot.q[1]*rot.q[2] -
	   2.0*rot.q[0]*rot.q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (2.0*rot.q[1]*rot.q[3] +
	   2.0*rot.q[0]*rot.q[2])*sp_matrix_get(z_coordinates,x,y);
	new_y =
	  (2.0*rot.q[1]*rot.q[2] +
	   2.0*rot.q[0]*rot.q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (rot.q[0]*rot.q[0] - rot.q[1]*rot.q[1] +
	   rot.q[2]*rot.q[2] - rot.q[3]*rot.q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (2.0*rot.q[2]*rot.q[3] -
	   2.0*rot.q[0]*rot.q[1])*sp_matrix_get(z_coordinates,x,y);
	new_z =
	  (2.0*rot.q[1]*rot.q[3] -
	   2.0*rot.q[0]*rot.q[2])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (2.0*rot.q[2]*rot.q[3] +
	   2.0*rot.q[0]*rot.q[1])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (rot.q[0]*rot.q[0] - rot.q[1]*rot.q[1] -
	   rot.q[2]*rot.q[2] + rot.q[3]*rot.q[3])*sp_matrix_get(z_coordinates,x,y);

	/* Round of the values nearest pixel. This function uses nearest
	   neighbour interpolation as oposed to the linear interpolation
	   used by the cuda code. */
	round_x = round((real)sp_3matrix_x(model)/2.0 - 0.5 + new_x);
	round_y = round((real)sp_3matrix_y(model)/2.0 - 0.5 + new_y);
	round_z = round((real)sp_3matrix_z(model)/2.0 - 0.5 + new_z);
	/* If the rotated coordinates are inside the extent of the model we
	   add the value to the model and 1 to the weight */
	if (round_x >= 0 && round_x < sp_3matrix_x(model) &&
	    round_y >= 0 && round_y < sp_3matrix_y(model) &&
	    round_z >= 0 && round_z < sp_3matrix_z(model)) {
	  sp_3matrix_set(model,round_x,round_y,round_z,
			 sp_3matrix_get(model,round_x,round_y,round_z)+w*sp_matrix_get(slice,x,y));
	  sp_3matrix_set(weight,round_x,round_y,round_z,sp_3matrix_get(weight,round_x,round_y,round_z)+w);
	}
      }// end of if
    }
  }
}

sp_matrix **read_images_cxi(const char *filename, const char *image_identifier, const char *mask_identifier,
			    const int number_of_images, const int image_side, const int binning_factor,
			    sp_imatrix **list_of_masks) {
  int status;
  hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) error_exit_with_message("Problem reading file %s", filename);
  hid_t dataset = H5Dopen1(file, image_identifier);
  if (dataset < 0) error_exit_with_message("Problem reading dataset %s in file %s", image_identifier, filename);

  hsize_t dims[3];
  hid_t file_dataspace = H5Dget_space(dataset);
  H5Sget_simple_extent_dims(file_dataspace, dims, NULL);
  if (number_of_images > dims[0]) error_exit_with_message("Dataset in %s does not contain %d images", filename, number_of_images);

  hsize_t hyperslab_start[3] = {0, 0, 0};
  hsize_t read_dims[3] = {number_of_images, dims[1], dims[2]};
  const int total_size = read_dims[0]*read_dims[1]*read_dims[2];
  status = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, hyperslab_start, NULL, read_dims, NULL);
  if (status < 0) error_exit_with_message("error selecting hyperslab in file %s", filename);

  hid_t data_dataspace = H5Screate_simple(3, read_dims, NULL);

  printf("read images\n");
  real *raw_image_data = malloc(total_size*sizeof(real));
  status = H5Dread(dataset, H5T_NATIVE_FLOAT, data_dataspace, file_dataspace, H5P_DEFAULT, raw_image_data);
  if (status < 0) error_exit_with_message("error reading data in file %s", filename);
  H5Dclose(dataset);

  dataset = H5Dopen1(file, mask_identifier);
  if (dataset < 0) error_exit_with_message("Problem reading dataset %s in file %s", mask_identifier, filename);

  printf("read mask\n");
  int *raw_mask_data = malloc(total_size*sizeof(int));
  status = H5Dread(dataset, H5T_NATIVE_INT, data_dataspace, file_dataspace, H5P_DEFAULT, raw_mask_data);
  H5Dclose(dataset);
  
  H5Sclose(data_dataspace);
  H5Sclose(file_dataspace);

  H5Fclose(file);

  /* Any pixels with values below zero are set to zero */
  for (int index = 0; index < total_size; index++) {
    if (raw_image_data[index] < 0.) {
      raw_image_data[index] = 0.;
    }
  }

  sp_matrix **list_of_images = malloc(number_of_images*sizeof(sp_matrix *));
  real pixel_sum, pixel_this;
  int mask_sum, mask_this;
  int transformed_x, transformed_y;
  int pixel_index;
  for (int image_index = 0; image_index < number_of_images; image_index++) {
    /* Allocate return arrays */
    list_of_images[image_index] = sp_matrix_alloc(image_side, image_side);
    list_of_masks[image_index] = sp_imatrix_alloc(image_side, image_side);

    /* Downsample images and masks by nested loops of all
       downsampled pixels and then all subpixels. */
    for (int x = 0; x < image_side; x++) {
      for (int y = 0; y < image_side; y++) {
	pixel_sum = 0.0;
	mask_sum = 0;
	/* Step through all sub-pixels and add up values */
	for (int xb = 0; xb < binning_factor; xb++) {
	  for (int yb = 0; yb < binning_factor; yb++) {
	    transformed_x = dims[2]/2 - (image_side/2)*binning_factor + x*binning_factor + xb;
	    transformed_y = dims[1]/2 - (image_side/2)*binning_factor + y*binning_factor + yb;
	    pixel_index = image_index*dims[1]*dims[2] + transformed_y*dims[2] + transformed_x;
	    if (transformed_x >= 0 && transformed_x < dims[2] &&
		transformed_y >= 0 && transformed_y < dims[1]) {
	      pixel_this = raw_image_data[pixel_index];
	      mask_this = raw_mask_data[pixel_index];
	    } else {
	      pixel_this = 0.;
	      mask_this = 0;
	    }
	    if (mask_this > 0) {
	      pixel_sum += pixel_this;
	      mask_sum += 1;
	    }
	  }
	}
	/* As long as there were at least one subpixel contributin to the
	   pixel (that is that was not masked out) we include the data and
	   don't mask out the pixel. */
	if (mask_sum > 0) {
	  sp_matrix_set(list_of_images[image_index], x, y, pixel_sum/(real)mask_sum);
	  sp_imatrix_set(list_of_masks[image_index], x, y, 1);
	} else {
	  sp_matrix_set(list_of_images[image_index], x, y, 0.);
	  sp_imatrix_set(list_of_masks[image_index], x, y, 0);
	}
      }
    }
  }
  return list_of_images;
}


/* Read images in spimage format and read the individual masks. The
   masks pointer should be allocated before calling, it is not
   allocated by this function. */
sp_matrix **read_images(Configuration conf, sp_imatrix **masks)
{
  sp_matrix **images = malloc(conf.number_of_images*sizeof(sp_matrix *));
  //masks = malloc(conf.number_of_images*sizeof(sp_imatrix *));
  Image *img;
  real *intensities = malloc(conf.number_of_images*sizeof(real));
  char filename_buffer[MAX_PATH_LENGTH];

  for (int i = 0; i < conf.number_of_images; i++) {
    intensities[i] = 1.0;
  }

  for (int i = 0; i < conf.number_of_images; i++) {
    sprintf(filename_buffer,"%s%.4d.h5", conf.image_prefix, i);
    img = sp_image_read(filename_buffer,0);

    /* Blur input image if specified in the configuration file. This
       might might be useful for noisy data if the noise is not taken
       into account by the diff_type. */
    /*
    if (conf.blur_image == 1) {
      Image *tmp = sp_gaussian_blur(img,conf.blur_image_sigma);
      sp_image_free(img);
      img = tmp;
    }
    */

    /* Allocate return arrays */
    images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
    masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);

    /* The algorithm can't handle negative data so if we have negative
       values they are simply set to 0. */
    for (int pixel_i = 0; pixel_i < sp_image_size(img); pixel_i++) {
      if (sp_real(img->image->data[pixel_i]) < 0.) {
	sp_real(img->image->data[pixel_i]) = 0.;
      }
    }

    /* Downsample images and masks by nested loops of all
       downsampled pixels and then all subpixels. */
    real pixel_sum, pixel_this;
    int mask_sum, mask_this;
    int transformed_x, transformed_y;
    for (int x = 0; x < conf.model_side; x++) {
      for (int y = 0; y < conf.model_side; y++) {
	pixel_sum = 0.0;
	mask_sum = 0;
	/* Step through all sub-pixels and add up values */
	for (int xb = 0; xb < conf.image_binning; xb++) {
	  for (int yb = 0; yb < conf.image_binning; yb++) {
	    transformed_x = sp_image_x(img)/2 - (conf.model_side/2)*conf.image_binning + x*conf.image_binning + xb;
	    transformed_y = sp_image_y(img)/2 - (conf.model_side/2)*conf.image_binning + y*conf.image_binning + yb;
	    if (transformed_x >= 0 && transformed_x < sp_image_x(img) &&
		transformed_y >= 0 && transformed_y < sp_image_y(img)) {
	      pixel_this = sp_cabs(sp_image_get(img, transformed_x, transformed_y, 0));
	      mask_this = sp_image_mask_get(img,  transformed_x, transformed_y, 0);
	    } else {
	      pixel_this = 0.;
	      mask_this = 0;
	    }
	    if (mask_this > 0) {
	      pixel_sum += pixel_this;
	      mask_sum += 1;
	    }
	  }
	}
	/* As long as there were at least one subpixel contributin to the
	   pixel (that is that was not masked out) we include the data and
	   don't mask out the pixel. */
	if (mask_sum > 0) {
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

/* Read the common mask in a similar way as in which the individual
   masks are read. The image values of the spimage file are used as
   the masks and the mask of the image is never read. Values of 0 are
   masked out. */
sp_imatrix *read_mask(Configuration conf)
{
  sp_imatrix *mask = sp_imatrix_alloc(conf.model_side,conf.model_side);;
  Image *mask_in = sp_image_read(conf.mask_file,0);
  /* read and rescale mask */
  int mask_sum;
  int this_value;
  int transformed_x, transformed_y;
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      mask_sum = 0;
      for (int xb = 0; xb < conf.image_binning; xb++) {
	for (int yb = 0; yb < conf.image_binning; yb++) {
	  transformed_x = sp_image_x(mask_in)/2 - (conf.model_side/2)*conf.image_binning + x*conf.image_binning + xb;
	  transformed_y = sp_image_y(mask_in)/2 - (conf.model_side/2)*conf.image_binning + y*conf.image_binning + yb;
	  if (transformed_x >= 0 && transformed_x < sp_image_x(mask_in) &&
	      transformed_y >= 0 && transformed_y < sp_image_y(mask_in)) {
	    this_value = sp_cabs(sp_image_get(mask_in, transformed_x, transformed_y, 0));
	  } else {
	    this_value = 0;
	  }
	  if (this_value) {
	    mask_sum += 1;
	  }
	}
      }
      if (mask_sum > 0) {
	sp_imatrix_set(mask,x,y,1);
      } else {
	sp_imatrix_set(mask,x,y,0);
      }
    }
  }
  sp_image_free(mask_in);

  /* Also mask out everything outside a sphere with diameter
     conf.model_side. In this way we avoid any bias from the
     alignment of the edges of the model. */
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

/* Normalize all diffraction patterns so that the average pixel
   value is 1.0 in each pattern. Use the common mask for the
   normalization. */
void normalize_images(sp_matrix **images, sp_imatrix **masks, Configuration conf)
{
  real sum, count;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
    sum = 0.;
    count = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 1) {
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

/* Normalize all diffraction patterns so that the average pixel value in
   each patterns in a circle of the specified radius is 0. An input radius
   of 0 means that the full image is used. */
void normalize_images_central_part(sp_matrix ** const images, sp_imatrix ** const masks, real radius, const Configuration conf) {
  const int x_max = conf.model_side;
  const int y_max = conf.model_side;
  /* If the radius is 0 we use the full image by setting the
     radius to a large number. */
  if (radius == 0) {
    radius = sqrt(pow(x_max, 2) + pow(y_max, 2))/2. + 2;
  }

  real *all_normalizations = malloc(conf.number_of_images*sizeof(real));
  
  /* Create a mask that marks the area to use. */
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

  /* Do the normalization using the mask just created. */
  real sum, count;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
    sum = 0.;
    count = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 1 && central_mask->data[i] == 1) {
	sum += images[i_image]->data[i];
	count += 1.;
      }
    }
    sum = (real) count / sum;
    all_normalizations[i_image] = sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
  char filename_buffer[MAX_PATH_LENGTH];
  sprintf(filename_buffer, "%s/normalization_factors.h5", conf.output_dir);
  write_1d_real_array_hdf5(filename_buffer, all_normalizations, conf.number_of_images);
}

/* Normalize all diffraction patterns so that the average pixel
   value in each pattern is 1.0. Use the individual masks for
   the normalization. */
void normalize_images_individual_mask(sp_matrix **images, sp_imatrix **masks,
				      Configuration conf)
{
  real sum, count;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
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

/* Normalize all diffraction patterns so that the average pixel value
   among all diffraction patterns is 1. This normalization is intended
   for when intensities are known and therefore the relative scaling
   must be preserved. */
void normalize_images_preserve_scaling(sp_matrix ** images, sp_imatrix **masks, Configuration conf) {
  int N_2d = conf.model_side*conf.model_side;
  real inner_sum = 0.;
  real outer_sum = 0.;
  real inner_count = 0.;
  real outer_count = 0.;
  for (int i = 0; i < N_2d; i++) {
    for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
      if (masks[i_image]->data[i] == 1) {
	inner_sum += images[i_image]->data[i];
	inner_count += 1.;
      }
    }
    if (inner_count > 0.) {
      outer_sum += inner_sum / inner_count;
      outer_count += 1.;
    }
  }
  real normalization_factor = (outer_count*(real)conf.number_of_images) / outer_sum;
  for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= normalization_factor;
    }
  }
}

/* This function writes some run information to a file and closes the
   file. This is intended to contain things that don't change during the
   run, currently number of images, the random seed and wether compact_output
   is used. This file is used by the viewer. */
void write_run_info(char *filename, Configuration conf, int random_seed) {
  hid_t file_id, space_id, dataset_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/number_of_images", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.number_of_images);
  H5Dclose(dataset_id);
  H5Sclose(space_id);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/compact_output", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.compact_output);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  
  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/random_seed", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &random_seed);
  H5Dclose(dataset_id);
  H5Sclose(space_id);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/recover_scaling", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.recover_scaling);
  H5Dclose(dataset_id);
  H5Sclose(space_id);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/wavelength", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.wavelength);
  H5Dclose(dataset_id);
  H5Sclose(space_id);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/pixel_size", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.pixel_size);
  H5Dclose(dataset_id);
  H5Sclose(space_id);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/detector_distance", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.detector_distance);
  H5Dclose(dataset_id);
  H5Sclose(space_id);

  H5Fclose(file_id);
}

/* The state file contains info about the run but opposed to the run info
   this contains things that change during the run. Currently it only
   contains the current iteration. This file is used by the viewer and
   is important to be able to view the data while the program is still
   running. */
hid_t open_state_file(char *filename) {
  hid_t file_id, space_id, dataset_id;
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  space_id = H5Screate(H5S_SCALAR);
  dataset_id = H5Dcreate1(file_id, "/iteration", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
  int iteration_start_value = -1;
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration_start_value);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  
  H5Fflush(file_id, H5F_SCOPE_GLOBAL);

  return file_id;
}

/* Write new information to the state file. Call this at the beginning of each iteration. */
void write_state_file_iteration(hid_t file_id, int iteration) {
  hid_t dataset_id;
  hsize_t file_size;
  H5Fget_filesize(file_id, &file_size);
  
  dataset_id = H5Dopen(file_id, "/iteration", H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration);
  H5Fflush(dataset_id, H5F_SCOPE_LOCAL);
  H5Dclose(dataset_id);
  H5Fflush(file_id, H5F_SCOPE_GLOBAL);
}

/* Close state file (duh). */
void close_state_file(hid_t file_id) {
  H5Fclose(file_id);
}

/* The rotation sampling is tabulated to save computational time (although i think
   the new code is fast enough so this is a bit ridicculus). This file reads the
   rotations from file and returns them and the weights in the respective input
   pointers and returns the number of rotations. */
int read_rotations_file(const char *filename, Quaternion **rotations, real **weights) {
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t dataset_id = H5Dopen1(file_id, "/rotations");
  hid_t space_id = H5Dget_space(dataset_id);
  hsize_t dims[2];
  hsize_t maxdims[2];
  H5Sget_simple_extent_dims(space_id, dims, maxdims);
  
  int N_slices = dims[0];
  
  real *input_array = malloc(N_slices*5*sizeof(real));
  H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_array);
  
  *rotations = malloc(N_slices*sizeof(Quaternion));
  *weights = malloc(N_slices*sizeof(real));
  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
    /*
    Quaternion *this_quaternion = quaternion_alloc();
    memcpy(this_quaternion->q, &input_array[i_slice*5], 4*sizeof(real));
    (*rotations)[i_slice] = this_quaternion;
    (*weights)[i_slice] = input_array[i_slice*5+4];
    */
    Quaternion this_quaternion;
    memcpy(this_quaternion.q, &input_array[i_slice*5], 4*sizeof(real));
    (*rotations)[i_slice] = this_quaternion;
    (*weights)[i_slice] = input_array[i_slice*5+4];
  }
  return N_slices;
}

/* Create a directory or do nothing if the directory already exists. This
   function also works if there are several levels of diretcories that does
   not exist. */
static void mkdir_recursive(const char *dir, int permission) {
  char tmp[MAX_PATH_LENGTH];
  char *p = NULL;
  size_t len;
  
  snprintf(tmp, sizeof(tmp), "%s", dir);
  len = strlen(tmp);
  if (tmp[len-1] == '/') {
    tmp[len-1] = 0;
  }
  for (p = tmp+1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      mkdir(tmp, permission);
      *p = '/';
    }
  }
  mkdir(tmp, permission);
}

/* Create the compressed model: model and the model
   weights used in the compress step: weight. */
static void create_initial_model_uniform(sp_3matrix *model, gsl_rng *rng) {
  const int N_model = sp_3matrix_size(model);
  for (int i = 0; i < N_model; i++) {
    model->data[i] = gsl_rng_uniform(rng);
  }
}

/* The model falls off raially in the same way as the average of
   all the patterns. On each pixel, some randomness is added with
   the strengh of conf.initial_modle_noise.*[initial pixel value] */
static void create_initial_model_radial_average(sp_3matrix *model, sp_matrix **images, const int N_images,
						sp_imatrix **masks, real initial_model_noise,
						gsl_rng *rng) {
  const int model_side = sp_3matrix_x(model);
  /* Setup for calculating radial average */
  real *radavg = malloc(model_side/2*sizeof(real));
  int *radavg_count = malloc(model_side/2*sizeof(int));
  int r;
  for (int i = 0; i < model_side/2; i++) {
    radavg[i] = 0.0;
    radavg_count[i] = 0;
  }
  
  /* Calculate the radial average */
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int x = 0; x < model_side; x++) {
      for (int y = 0; y < model_side; y++) {
	if (sp_imatrix_get(masks[i_image], x, y) > 0 && sp_matrix_get(images[i_image], x, y) >= 0.) {
	  //if (sp_matrix_get(images[i_image], x, y) >= 0.) {
	  r = (int)sqrt(pow((real)x - model_side/2.0 + 0.5,2) +
			pow((real)y - model_side/2.0 + 0.5,2));
	  if (r < model_side/2.0) {
	    radavg[r] += sp_matrix_get(images[i_image],x,y);
	    radavg_count[r] += 1;
	  }
	}
      }
    }
  }
  for (int i = 0; i < model_side/2; i++) {
    if (radavg_count[i] > 0) {
      radavg[i] /= (real) radavg_count[i];
    } else {
      radavg[i] = -1.0;
    }
  }
  /* Fill the model and apply noise */
  real rad;
  for (int x = 0; x < model_side; x++) {
    for (int y = 0; y < model_side; y++) {
      for (int z = 0; z < model_side; z++) {
	rad = sqrt(pow((real)x - model_side/2.0 + 0.5,2) +
		   pow((real)y - model_side/2.0 + 0.5,2) +
		   pow((real)z - model_side/2.0 + 0.5,2));
	r = (int)rad;

	if (radavg[r] == -1) {
	  sp_3matrix_set(model, x, y, z, -1.0);
	} else if (r < model_side/2-1) {
	  sp_3matrix_set(model, x, y, z, (radavg[r]*(1.0 - (rad - (real)r)) +
					  radavg[r+1]*(rad - (real)r)) * (1. + initial_model_noise*gsl_rng_uniform(rng)));
	} else if (r < model_side/2){
	  sp_3matrix_set(model, x, y, z, radavg[r] * (1. + initial_model_noise*gsl_rng_uniform(rng)));
	} else {
	  sp_3matrix_set(model, x, y, z, -1.0);
	}
      }
    }
  }
}

/* Assemble the model from the given diffraction patterns
   with a random orientation assigned to each. */
static void create_initial_model_random_orientations(sp_3matrix *model, sp_3matrix *weight, sp_matrix **images,
						     const int N_images, sp_imatrix **masks, sp_matrix *x_coordinates,
						     sp_matrix *y_coordinates, sp_matrix *z_coordinates, gsl_rng *rng) {
  const int N_model = sp_3matrix_size(model);
  Quaternion random_rot;
  for (int i_image = 0; i_image < N_images; i_image++) {
    random_rot = quaternion_random(rng);
    insert_slice(model, weight, images[i_image], masks[i_image], 1.0, random_rot,
		 x_coordinates, y_coordinates, z_coordinates);
    //free(random_rot);
  }
  for (int i = 0; i < N_model; i++) {
    if (weight->data[i] > 0.0) {
      model->data[i] /= (weight->data[i]);
    } else {
      model->data[i] = 0.0;
    }
  }
}

/* Assemble the model from the given diffraction patterns
   with a rotation assigned from the file provided in
   conf.initial_rotations_file. */
static void create_initial_model_given_orientations(sp_3matrix *model, sp_3matrix *weight, sp_matrix **images, const int N_images, 
						    sp_imatrix **masks, sp_matrix *x_coordinates, sp_matrix *y_coordinates,
						    sp_matrix *z_coordinates, const char *init_rotations_file) {
  const int N_model = sp_3matrix_size(model);
  FILE *given_rotations_file = fopen(init_rotations_file, "r");
  //Quaternion *this_rotation = quaternion_alloc();
  Quaternion this_rotation;
  for (int i_image = 0; i_image < N_images; i_image++) {
    fscanf(given_rotations_file, "%g %g %g %g\n", &(this_rotation.q[0]), &(this_rotation.q[1]), &(this_rotation.q[2]), &(this_rotation.q[3]));
    insert_slice(model, weight, images[i_image], masks[i_image], 1., this_rotation,
		 x_coordinates, y_coordinates, z_coordinates);
  }
  //free(this_rotation);
  fclose(given_rotations_file);
    
  for (int i = 0; i < N_model; i++) {
    if (weight->data[i] > 0.) {
      model->data[i] /= weight->data[i];
    }else {
      model->data[i] = -1.;
    }
  }
}

/* Read the initial model from file.*/
static void create_initial_model_file(sp_3matrix *model, const char *model_file) {
  const int N_model = sp_3matrix_size(model);
  Image *model_in = sp_image_read(model_file,0);
  if (sp_3matrix_x(model) != sp_image_x(model_in) ||
      sp_3matrix_y(model) != sp_image_y(model_in) ||
      sp_3matrix_z(model) != sp_image_z(model_in))
    error_exit_with_message("Input model is of wrong size. Should be (%i, %i, %i\n",
		      sp_3matrix_x(model), sp_3matrix_y(model), sp_3matrix_z(model));
  for (int i = 0; i < N_model; i++) {
    model->data[i] = sp_cabs(model_in->image->data[i]);
  }
  sp_image_free(model_in);
}

int main(int argc, char **argv)
{
  /* Parse command-line options */
  char configuration_filename[MAX_PATH_LENGTH] = "emc.conf";
  int chosen_device = -1; // negative numbers means the program chooses automatically
  char help_text[] =
    "Options:\n\
-h Show this text\n\
-c CONFIGURATION_FILE Specify a configuration file";
  int command_line_conf;
  while ((command_line_conf = getopt(argc, argv, "hc:d:")) != -1) {
    if (command_line_conf == -1) {
      break;
    }
    switch(command_line_conf) {
    case ('h'):
      printf("%s\n", help_text);
      exit(0);
      break;
    case ('c'):
      strcpy(configuration_filename, optarg);
      break;
    case('d'):
      chosen_device = atoi(optarg);
      int number_of_devices = cuda_get_number_of_devices();
      if (chosen_device >= number_of_devices) {
	printf("Asking for device %i with only %i devices available\n", chosen_device, number_of_devices);
	exit(0);
      }
      break;
    }
  }
  
  /* Capture a crtl-c event to make a final iteration be
   run with the individual masked used in the compression.
   This is consistent with the final iteration when not
   interupted. Ctrl-c again will exit immediatley. */
  signal(SIGINT, nice_exit);

  /* Set the cuda device */
  if (chosen_device >= 0) {
    cuda_set_device(chosen_device);
  } else {
    cuda_choose_best_device();
  }
  cuda_print_device_info();

  /* This buffer is used for names of all output files */
  char filename_buffer[MAX_PATH_LENGTH];
  
  /* Read the configuration file */
  Configuration conf;
  init_configuration(&conf);
  create_default_config(&conf);
  int conf_return = read_configuration_file(configuration_filename, &conf);
  if (conf_return == 0)
    error_exit_with_message("Can't read configuration file %s\nRun emc -h for help.", configuration_filename);
  sprintf(filename_buffer, "%sout", configuration_filename);
  write_configuration_file(filename_buffer, &conf);

  /* Create the output directory if it does not exist. */
  mkdir_recursive(conf.output_dir, 0777);

  /* Create constant versions of some of the commonly used
     variables from the configuration file. Also create some
     useful derived variables */
  const int N_images = conf.number_of_images;
  const int slice_chunk = conf.chunk_size;
  const int N_2d = conf.model_side*conf.model_side;
  const int N_model = conf.model_side*conf.model_side*conf.model_side;

  int N_images_included;
  if (conf.calculate_r_free) {
    N_images_included = (1.-conf.r_free_ratio) * (float) N_images;
  } else {
    N_images_included = N_images;
  }

  /* Read the list of sampled rotations and rotational weights */
  //Quaternion **rotations;
  Quaternion *rotations;
  real *weights;
  const int N_slices = read_rotations_file(conf.rotations_file, &rotations, &weights);
  /*
  printf("start generating rotations\n");
  clock_t begin, end;
  begin = clock();
  const int N_slices = generate_rotation_list(20, &rotations, &weights);
  end = clock();
  printf("done generating rotations: %g s\n", (real) (end-begin) / CLOCKS_PER_SEC);
  */

  /* Copy rotational weights to the GPU */
  real *d_weights;
  cuda_allocate_real(&d_weights, N_slices);
  cuda_copy_real_to_device(weights, d_weights, N_slices);
  
  /* Get a random seed from /dev/random or from the
     configuration file if provided. */
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  unsigned long int seed = 0;
  if(conf.random_seed == 0){
    FILE *devrandom;
    if ((devrandom = fopen("/dev/random","r")) == NULL) {
      fprintf(stderr,"Cannot open /dev/random, setting seed to 0\n");
      seed = 0;
    } else {
      fread(&seed,sizeof(seed),1,devrandom);
      fclose(devrandom);
    }
  }else{
    seed = conf.random_seed;
  }
  srand(seed);
  gsl_rng_set(rng, rand());

  /* Write run_info.h5. This file contains some information about
     the setup of the run that are useful to the viewer or anyone
     looking at the data. */
  sprintf(filename_buffer, "%s/run_info.h5", conf.output_dir);
  write_run_info(filename_buffer, conf, seed);

  /* Create the state.h5 file. This file contains run information
     that changes throughout the run and is continuously updated
     and is used by for example the viewer when viewing the output
     from an unfinished run.*/
  sprintf(filename_buffer, "%s/state.h5", conf.output_dir);
  hid_t state_file = open_state_file(filename_buffer);

  /* Read images and mask */
  sp_imatrix **individual_masks = malloc(conf.number_of_images*sizeof(sp_imatrix *));
  sp_matrix **images = read_images(conf, individual_masks);
  /*
  sp_matrix **images = read_images_cxi("/home/ekeberg/Data/LCLS_SPI/2015July/cxi/narrow_filter_normalized.cxi",
				       "/entry_1/data", "/entry_1/mask", conf.number_of_images, conf.model_side,
				       conf.image_binning, masks);
  */

  sp_imatrix ** masks;
  sp_imatrix *common_mask;
  sp_imatrix **common_masks;
  if (conf.individual_masks == 0) {
    common_mask = read_mask(conf);
    common_masks = malloc(N_images*sizeof(sp_imatrix));
    for (int i_image = 0; i_image < N_images; i_image++) {
      common_masks[i_image] = sp_imatrix_alloc(conf.model_side, conf.model_side);
      sp_imatrix_memcpy(common_masks[i_image], common_mask);
    }
    masks = common_masks;
  } else {
    masks = individual_masks;
  }
  
  if (conf.normalize_images) {
    if (!conf.recover_scaling) {
      normalize_images_preserve_scaling(images, masks, conf);
    } else {
      //real central_part_radius = 0.; // Zero here means that the entire image is used.
      real central_part_radius = 10.; // Zero here means that the entire image is used.
      normalize_images_central_part(images, masks, central_part_radius, conf);
    }
  }

  /* Write preprocessed images 1: Calculate the maximum of all
     images. This is used for normalization of the outputed pngs.*/
  real image_max = 0.;
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 1 && images[i_image]->data[i] > image_max) {
	image_max = images[i_image]->data[i];
      }
    }
  }

  /* Write preprocessed images 2: Create the spimage images to
     output. Including putting in the general mask.*/
  Image *write_image = sp_image_alloc(conf.model_side, conf.model_side, 1);
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i]) {
	sp_real(write_image->image->data[i]) = images[i_image]->data[i];
      } else {
	sp_real(write_image->image->data[i]) = 0.0;
      }
      write_image->mask->data[i] = masks[i_image]->data[i];
    }
    sprintf(filename_buffer, "%s/image_%.4d.h5", conf.output_dir, i_image);
    sp_image_write(write_image, filename_buffer, 0);

    /* Set a corner pixel to image_max. This assures correct relative scaling of pngs. */
    write_image->image->data[0] = sp_cinit(image_max, 0.);
    sprintf(filename_buffer, "%s/image_%.4d.png", conf.output_dir, i_image);
    sp_image_write(write_image, filename_buffer, SpColormapJet|SpColormapLogScale);
  }
  sp_image_free(write_image);

  /* Precalculate coordinates. These coordinates represent an
     Ewald sphere with the xy plane with liftoff in the z direction.
     These coordinates then only have to be rotated to get coordinates
     for expansion and compression. */
  sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength,
			x_coordinates, y_coordinates, z_coordinates);

  /* sprintf(filename_buffer, "%s/coordinates_x.h5", conf.output_dir); */
  /* write_2d_real_array_hdf5(filename_buffer, x_coordinates->data, conf.model_side, conf.model_side); */
  /* sprintf(filename_buffer, "%s/coordinates_y.h5", conf.output_dir); */
  /* write_2d_real_array_hdf5(filename_buffer, y_coordinates->data, conf.model_side, conf.model_side); */
  /* sprintf(filename_buffer, "%s/coordinates_z.h5", conf.output_dir); */
  /* write_2d_real_array_hdf5(filename_buffer, z_coordinates->data, conf.model_side, conf.model_side); */
  /* exit(1); */


  /* Create the compressed model: model and the model
     weights used in the compress step: weight.*/
  sp_3matrix *model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  sp_3matrix *weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  for (int i = 0; i < N_model; i++) {
    model->data[i] = 0.0;
    weight->data[i] = 0.0;
  }

  /* Initialize the model in the way specified by
     the configuration file.*/
  if (conf.initial_model == initial_model_uniform) {
    /* Set every pixel in the model to a random number between
       zero and one (will be normalized later) */
    create_initial_model_uniform(model, rng);
  } else if (conf.initial_model == initial_model_radial_average) {
    /* The model falls off raially in the same way as the average of
     all the patterns. On each pixel, some randomness is added with
     the strengh of conf.initial_modle_noise.*[initial pixel value] */
    create_initial_model_radial_average(model, images, N_images, masks, conf.initial_model_noise, rng);
  } else if (conf.initial_model == initial_model_random_orientations) {
    /* Assemble the model from the given diffraction patterns
       with a random orientation assigned to each. */
    create_initial_model_random_orientations(model, weight, images, N_images, masks, x_coordinates,
					     y_coordinates, z_coordinates, rng);
  }else if (conf.initial_model == initial_model_file) {
    /* Read the initial model from file.*/
    create_initial_model_file(model, conf.initial_model_file);
  } else if (conf.initial_model == initial_model_given_orientations) {
    /* Assemble the model from the given diffraction patterns
       with a rotation assigned from the file provided in
       conf.initial_rotations_file. */
    create_initial_model_given_orientations(model, weight, images, N_images, masks, x_coordinates,
					    y_coordinates, z_coordinates, conf.initial_rotations_file);
  }

  /* Allocate spimage object used for outputting the model.*/
  Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
  for (int i = 0; i < N_model; i++) {
    //if (weight->data[i] > 0.0) {
    if (model->data[i] >= 0.0) {
      model_out->image->data[i] = sp_cinit(model->data[i],0.0);
      model_out->mask->data[i] = 1;
    } else {
      model_out->image->data[i] = sp_cinit(0., 0.);
      model_out->mask->data[i] = 0;
    }
  }

  /* Write the initial model. */
  sprintf(filename_buffer,"%s/model_init.h5", conf.output_dir);
  sp_image_write(model_out,filename_buffer,0);
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(weight->data[i],0.0);
  }
  sprintf(filename_buffer,"%s/model_init_weight.h5", conf.output_dir);
  sp_image_write(model_out,filename_buffer,0);

  /* Create the matrix radius where the value of each pixel
     is the distance to the center. */
  sp_matrix *radius = sp_matrix_alloc(conf.model_side,conf.model_side);
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
	sp_matrix_set(radius,x,y,sqrt(pow((real) x - conf.model_side/2.0 + 0.5, 2) +
				      pow((real) y - conf.model_side/2.0 + 0.5, 2)));
    }
  }

  /* Create and initialize the scaling variables on the CPU. */
  real *scaling = malloc(N_images*N_slices*sizeof(real));
  for (int i = 0; i < N_images*N_slices; i++) {
    scaling[i] = 1.0;
  }

  /* Create active images and initialize them. If calculate_r_free
     is used the active_images variable keeps track of which
     images are included an which are excluded. */
  int *active_images = malloc(N_images*sizeof(int));
  for (int i_image = 0; i_image < N_images; i_image++) {
    active_images[i_image] = 1;
  }
  if (conf.calculate_r_free) {
    int *index_list = malloc(N_images*sizeof(int));
    for (int i_image = 0; i_image < N_images; i_image++) {
      index_list[i_image] = i_image;
    }
    gsl_ran_shuffle(rng, index_list, N_images, sizeof(int));
    int cutoff = N_images - N_images_included;
    for (int i = 0; i < cutoff; i++) {
      active_images[index_list[i]] = -1;
    }
    free(index_list);
  }


  /* Create responsability matrix on the CPU and associated
     variables.*/
  real *respons = malloc(N_slices*N_images*sizeof(real));
  real total_respons;
  real *average_resp = malloc(N_slices*sizeof(real));

  /* Create and initialize GPU variables. */
  /* Expanded model. Does normally not fit the entire model
     because memory is limited. Therefore it is always used
     in chunks. */
  real * d_slices;
  cuda_allocate_slices(&d_slices,conf.model_side,slice_chunk);

  /* Model. This exists in two versions so that the current model
     and the model from the previous iteration can be compared. */
  real * d_model;
  real * d_model_updated;
  real * d_model_tmp;
  cuda_allocate_model(&d_model,model);
  cuda_allocate_model(&d_model_updated,model);
  if (conf.recover_scaling) {
    cuda_normalize_model(model, d_model);
    cuda_normalize_model(model, d_model_updated);
  }

  /* Model weight. */
  real * d_weight;
  cuda_allocate_model(&d_weight,weight);

  /* List of all sampled rotations. Used in both expansion and
     compression. Does not change. */
  real * d_rotations;
  cuda_allocate_rotations(&d_rotations, rotations, N_slices);

  /* Precalculated Ewald sphere. */
  real * d_x_coord;
  real * d_y_coord;
  real * d_z_coord;
  cuda_allocate_coords(&d_x_coord,
		       &d_y_coord,
		       &d_z_coord,
		       x_coordinates,
		       y_coordinates, 
		       z_coordinates);

  /* The 2D mask used for all diffraction patterns */
  /* int * d_mask; */
  /* cuda_allocate_mask(&d_mask, mask); */

  /* Array of all diffraction patterns. */
  real *d_images;
  cuda_allocate_images(&d_images, images, N_images);
  /* real * d_images_common_mask; */
  /* cuda_allocate_images(&d_images_common_mask, images, N_images); */
  /* cuda_apply_single_mask(d_images_common_mask, d_mask, N_2d, N_images); */

  /* Individual masks read from each diffraction pattern.
     Only used for the last iteration. */
  int * d_masks;
  cuda_allocate_individual_masks(&d_masks, masks, N_images);
  int * d_individual_masks;
  cuda_allocate_individual_masks(&d_individual_masks, individual_masks, N_images);

  /* Array of all diffraction patterns with mask applied. */
  real * d_images_individual_mask;
  cuda_allocate_images(&d_images, images, N_images);
  cuda_apply_masks(d_images, d_masks, N_2d, N_images);

  cuda_allocate_images(&d_images_individual_mask, images, N_images);
  cuda_apply_masks(d_images_individual_mask, d_individual_masks, N_2d, N_images);

  /* real *d_images; */
  /* if (conf.individual_masks) { */
  /*   d_images = d_images_individual_mask; */
  /* } else { */
  /*   d_images = d_images_common_mask; */
  /* } */
  
  /* Responsability matrix */
  real * d_respons;
  cuda_allocate_real(&d_respons, N_slices*N_images);

  /* Scaling */
  real * d_scaling;
  cuda_allocate_scaling_full(&d_scaling, N_images, N_slices);

  /* Weighted power is an internal variable in the EMC
     algorithm. Never exists on the CPU. */
  real *d_weighted_power;
  cuda_allocate_real(&d_weighted_power,N_images);

  /* The fit is a measure of how well the data matches the
     model that is more intuitive than the likelihood since
     it is in the [0, 1] range. */
  real *fit = malloc(N_images*sizeof(real));
  real *d_fit;
  cuda_allocate_real(&d_fit,N_images);

  /* fit_best_rot is like fit but instead ov weighted average
     over all orientations, each image is just considered in its
     best fitting orientation. */
  real *fit_best_rot = malloc(N_images*sizeof(real));
  real *d_fit_best_rot;
  cuda_allocate_real(&d_fit_best_rot, N_images);

  /* If calculate_r_free is used the active_images variable keeps
     track of which images are included an which are excluded. */
  int *d_active_images;
  cuda_allocate_int(&d_active_images,N_images);
  cuda_copy_int_to_device(active_images, d_active_images, N_images);

  /* 3D array where each value is the distance to the center
     of that pixel. */
  real *d_radius;
  cuda_allocate_real(&d_radius, N_2d);
  cuda_copy_real_to_device(radius->data, d_radius, N_2d);

  /* Radial fit is the same as fit but instead of as a function
     of diffraction pattern index it is presented as a function
     of distance to the center. */
  real *radial_fit = malloc(conf.model_side/2*sizeof(real));
  real *radial_fit_weight = malloc(conf.model_side/2*sizeof(real));
  real *d_radial_fit;
  real *d_radial_fit_weight;
  cuda_allocate_real(&d_radial_fit, conf.model_side/2);
  cuda_allocate_real(&d_radial_fit_weight, conf.model_side/2);

  /* best_rotation stores the index of the rotation with the
     highest responsability for each diffraction pattern. */
  int *best_rotation = malloc(N_images*sizeof(int));
  int *d_best_rotation;
  cuda_allocate_int(&d_best_rotation, N_images);
  
  /* Open files that will be continuously written to during execution. */
  sprintf(filename_buffer, "%s/likelihood.data", conf.output_dir);
  FILE *likelihood = fopen(filename_buffer, "wp");
  sprintf(filename_buffer, "%s/best_rot.data", conf.output_dir);
  FILE *best_rot_file = fopen(filename_buffer, "wp");
  FILE *best_quat_file;
  sprintf(filename_buffer, "%s/fit.data", conf.output_dir);
  FILE *fit_file = fopen(filename_buffer,"wp");
  sprintf(filename_buffer, "%s/fit_best_rot.data", conf.output_dir);
  FILE *fit_best_rot_file = fopen(filename_buffer,"wp");
  sprintf(filename_buffer, "%s/radial_fit.data", conf.output_dir);
  FILE *radial_fit_file = fopen(filename_buffer,"wp");
  FILE *r_free;
  if (conf.calculate_r_free) {
    sprintf(filename_buffer, "%s/r_free.data", conf.output_dir);
    r_free = fopen(filename_buffer, "wp");
  }
  /* This scaling output is for the scaling for the best fitting
     orientation for each diffraction pattern. This is used by
     the viewer. */
  hid_t scaling_file;
  sprintf(filename_buffer, "%s/best_scaling.h5", conf.output_dir);
  hid_t scaling_dataset;
  if (conf.recover_scaling) {
    scaling_dataset = init_scaling_file(filename_buffer, N_images, &scaling_file);
  }

  /* The weightmap allows to add a radially changing weight
     to pixels. This is normally not used though and requires
     recompilation to turn on. */
  real *d_weight_map;
  cuda_allocate_weight_map(&d_weight_map, conf.model_side);
  real weight_map_radius, weight_map_falloff;
  real weight_map_radius_start = conf.model_side; // Set start radius to contain entire pattern
  real weight_map_radius_final = conf.model_side; // Set final radius to contain entire pattern

  /* Create variables used in the main loop. */
  real sigma;
  int current_chunk;
  
  clock_t start_time, end_time;

  /* Start the main EMC loop */
  for (int iteration = 0; iteration < conf.number_of_iterations; iteration++) {
    start_time = clock();
    /* If ctrl-c was pressed execution stops but the final
       iteration using individual masks and cleenup still runs. */
    if (quit_requested == 1) {
      break;
    }
    printf("\niteration %d\n", iteration);

    /* Sigma is a variable that describes the noise that is
       typically either constant or decreasing on every iteration. */
    //sigma = conf.sigma_final + (conf.sigma_start-conf.sigma_final)*exp(-iteration/(float)conf.sigma_half_life*log(2.));
    if (iteration <= conf.sigma_half_life) {
      sigma = conf.sigma_start;
    } else {
      sigma = conf.sigma_final;
    }
    printf("sigma = %g\n", sigma);

    /* Calculate the weightmap radius for this particular iteration. */
    weight_map_radius = weight_map_radius_start + ((weight_map_radius_final-weight_map_radius_start) *
						   ((real)iteration / ((real)conf.sigma_half_life)));
    weight_map_falloff = 0.;
    /* This function sets the d_weight_map to all ones. Other
       functions are available in emc_cuda.cu.*/
    //cuda_allocate_weight_map(&d_weight_map, conf.model_side);

    /* Reset the fit parameters */
    int radial_fit_n = 1; // Allow less frequent output of the fit by changing this output period
    printf("set_to_zero d_fit\n");
    cuda_set_to_zero(d_fit,N_images);
    printf("set_to_zero d_radial_fit\n");
    cuda_set_to_zero(d_radial_fit,conf.model_side/2);
    printf("set_to_zero d_radial_fit_weight\n");
    cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2);
    printf("no more set_to_zero\n");

    /* Find and output the best orientation for each diffraction pattern,
       i.e. the one with the highest responsability. */
    cuda_calculate_best_rotation(d_respons, d_best_rotation, N_images, N_slices);
    cuda_copy_int_to_host(best_rotation, d_best_rotation, N_images);

    /* First output best orientation as an index in one big file
       containing all iterations. */
    for (int i_image = 0; i_image < N_images; i_image++) {
      fprintf(best_rot_file, "%d ", best_rotation[i_image]);
    }
    fprintf(best_rot_file, "\n");
    fflush(best_rot_file);

    /* Then output best orientation as a quaternion in a new
       file for each iteration. */
    sprintf(filename_buffer, "%s/best_quaternion_%.4d.data", conf.output_dir, iteration);
    best_quat_file = fopen(filename_buffer, "wp");
    for (int i_image = 0; i_image < N_images; i_image++) {
      fprintf(best_quat_file, "%g %g %g %g\n", rotations[best_rotation[i_image]].q[0], rotations[best_rotation[i_image]].q[1],
	      rotations[best_rotation[i_image]].q[2], rotations[best_rotation[i_image]].q[3]);
    }
    fclose(best_quat_file);
	
    /* In this loop through the chunks the "fit" is calculated */

    /* DEBUG */
    /* printf("allocate slices\n"); */
    /* real *slices = malloc(N_slices*N_2d*sizeof(real)); // <debug> */
    /* DEBUG END */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord, slice_start, current_chunk);

      /*
      printf("copy slices ot array (%d / %d)\n", 1+slice_start/slice_chunk, N_slices/slice_chunk);
      cuda_copy_real_to_host(&(slices[slice_start*N_2d]), d_slices, current_chunk*N_2d); // <debug>
      */
      /* Calculate the "fit" between the diffraction patterns
	 and the compressed model. There are two versions of this:
	 one weighted average and one where only the best orientation
	 of each pattern is considered. */
      cuda_calculate_fit(d_slices, d_images, d_masks, d_scaling,
			 d_respons, d_fit, sigma, N_2d, N_images,
			 slice_start, current_chunk);
      cuda_calculate_fit_best_rot(d_slices, d_images, d_masks, d_scaling,
				  d_best_rotation, d_fit_best_rot, N_2d, N_images,
				  slice_start, current_chunk);
      /* Calculate a radially averaged version of the weightd "fit" */
      if (iteration % radial_fit_n == 0 && iteration != 0) {
	cuda_calculate_radial_fit(d_slices, d_images, d_masks,
				  d_scaling, d_respons, d_radial_fit,
				  d_radial_fit_weight, d_radius,
				  N_2d, conf.model_side, N_images, slice_start,
				  current_chunk);
      }
      
    }
    /*
    printf("print file\n");
    sprintf(filename_buffer, "%s/debug_slices_%.4d.h5", conf.output_dir, iteration);
    write_3d_array_hdf5(filename_buffer, slices, N_slices, conf.model_side, conf.model_side);
    free(slices);
    */
    /* Output the fits */
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

    /* Output the radial fit if it is calculated */
    if ((iteration % radial_fit_n == 0 && iteration != 0)) {
      /* The radial average needs to be normalized on the CPU before it is output. */
      cuda_copy_real_to_host(radial_fit, d_radial_fit, conf.model_side/2);
      cuda_copy_real_to_host(radial_fit_weight, d_radial_fit_weight, conf.model_side/2);
      for (int i = 0; i < conf.model_side/2; i++) {
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
    /* This is the end of the "fit" calculation and output. */

    /* In this loop through the chunks the scaling is updated.
       This only runs if the user has specified that the intensity
       is unknown. */
    if (conf.recover_scaling) {
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	cuda_update_scaling_full(d_images, d_slices, d_masks, d_scaling, d_weight_map, N_2d, N_images, slice_start, current_chunk, conf.diff);
      }

      /* Output scaling */
      cuda_copy_real_to_host(scaling, d_scaling, N_images*N_slices);

      /* Only output scaling and responsabilities if compact_output
	 is turned off. */
      if (conf.compact_output == 0) {
	sprintf(filename_buffer, "%s/scaling_%.4d.h5", conf.output_dir, iteration);
	write_2d_real_array_hdf5(filename_buffer, scaling, N_slices, N_images);
      }
      
      /* Output the best scaling */
      cuda_copy_real_to_host(respons, d_respons, N_slices*N_images);
      write_scaling_to_file(scaling_dataset, iteration, scaling, respons, N_slices);
    }
    /* This is the end of the scaling update. */

    /* In this loop through the chunks the responsabilities are
       updated. */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      
      cuda_get_slices(model,d_model,d_slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);

      cuda_calculate_responsabilities(d_slices, d_images, d_masks, d_weight_map,
				      sigma, d_scaling, d_respons, d_weights, 
				      N_2d, N_images, slice_start,
				      current_chunk, conf.diff);

    }

    /* Calculate R-free. Randomly (earlier) select a number of images
       that are excluded from the compress step. These are still compared
       present in the responsability matrix though and this value is
       used as an indication whether tha algorithm is overfitting
       or not. */
    if (conf.calculate_r_free) {
      cuda_copy_real_to_host(respons, d_respons, N_slices*N_images);

      /* Calculate the best responsability for each diffraction pattern. */
      real *best_respons = malloc(N_images*sizeof(real));
      real this_respons;
      for (int i_image = 0; i_image < N_images; i_image++) {
	best_respons[i_image] = respons[0*N_images+i_image];
	for (int i_slice = 1; i_slice < N_slices; i_slice++) {
	  this_respons = respons[i_slice*N_images+i_image];
	  if (this_respons > best_respons[i_image]) {
	    best_respons[i_image] = this_respons;
	  }
	}
      }

      /* Calculate the highest responsability for any diffraction pattern. */
      real universal_best_respons = best_respons[0];
      for (int i_image = 1; i_image < N_images; i_image++) {
	this_respons = best_respons[i_image];
	if (this_respons > universal_best_respons) {
	  universal_best_respons = this_respons;
	}
      }

      /* Take responsability from log to real space. */
      int range = (int) (universal_best_respons / log(10.));
      for (int i_image = 0; i_image < N_images; i_image++) {
	best_respons[i_image] = exp(best_respons[i_image]-range*log(10));
      }

      /* Sum up the best responsabilities for the included and the
	 excluded diffraction patterns respectively. */
      real average_best_response_included = 0.;
      real average_best_response_free = 0.;
      int included_count = 0;
      int free_count = 0;
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (active_images[i_image] == 1) {
	  average_best_response_included += best_respons[i_image];
	  included_count++;
	} else if (active_images[i_image] == -1) {
	  average_best_response_free += best_respons[i_image];
	  free_count++;
	}
      }
      free(best_respons);
      average_best_response_included /= (float) included_count;
      average_best_response_free /= (float) free_count;
      
      /* Write to file the average best responsability for both the
	 included and excluded diffraction patterns */
      fprintf(r_free, "%g %g %d\n",  average_best_response_included, average_best_response_free, range);
      fflush(r_free);
    }

    /* Normalize responsabilities. */
    cuda_calculate_responsabilities_sum(respons, d_respons, N_slices, N_images);
    /* DEBUG */
    /* cuda_copy_real_to_host(respons, d_respons, N_slices*N_images); */
    /* sprintf(filename_buffer, "%s/responsabilities_before_norm_%.4d.h5", conf.output_dir, iteration); */
    /* write_2d_real_array_hdf5_transpose(filename_buffer, respons, N_slices, N_images); */
    /* END DEBUG */
    cuda_normalize_responsabilities(d_respons, N_slices, N_images);
    cuda_copy_real_to_host(respons, d_respons, N_slices*N_images);

    /* Output responsabilities. Only output scaling and
       responsabilities if compact_output is turned off. */
    if (conf.compact_output == 0) {
      sprintf(filename_buffer, "%s/responsabilities_%.4d.h5", conf.output_dir, iteration);
      write_2d_real_array_hdf5_transpose(filename_buffer, respons, N_slices, N_images);
    }

    /* Output average responsabilities. These are plotted in the
       viewer. */
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      average_resp[i_slice] = 0.;
    }
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
      for (int i_image = 0; i_image < N_images; i_image++) {
	average_resp[i_slice] += respons[i_slice*N_images+i_image];
      }
    }
    sprintf(filename_buffer, "%s/average_resp_%.4d.h5", conf.output_dir, iteration);
    write_1d_real_array_hdf5(filename_buffer, average_resp, N_slices);

    /* Calculate and output likelihood, which is the sum of all
       responsabilities. */
    total_respons = cuda_total_respons(d_respons,respons,N_images*N_slices);
    fprintf(likelihood,"%g\n",total_respons);
    fflush(likelihood);


    /* Exclude images. Use the assumption that the diffraction patterns
       with the lowest maximum responsability does not belong in the data.
       Therefore these are excluded from the compression step. */
    real *best_respons;
    if (conf.exclude_images == 1 && iteration > -1) {
      /* Find the highest responsability for each diffraction pattern */
      best_respons = malloc(N_images*sizeof(real));
      for (int i_image = 0; i_image < N_images; i_image++) {
	best_respons[i_image] = respons[0*N_images+i_image];
	for (int i_slice = 1; i_slice < N_slices; i_slice++) {
	  if (!isnan(respons[i_slice*N_images+i_image]) && (respons[i_slice*N_images+i_image] > best_respons[i_image] || isnan(best_respons[i_image]))) {
	    best_respons[i_image] = respons[i_slice*N_images+i_image];
	  }
	}
	
	/* Check for nan in responsabilities. */
	if (isnan(best_respons[i_image])) {
	  printf("%d: best resp is nan\n", i_image);
	  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	    if (!isnan(respons[i_slice*N_images+i_image])) {
	      printf("tot resp is bad but single is good\n");
	    }
	  }
	}
      }

      /* Create a new best respons array to be sorted. This one only
	 contains the diffraction patterns not excluded by the R-free
	 calculation. */
      real *best_respons_copy = malloc(N_images_included*sizeof(real));
      int count = 0;
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (active_images[i_image] >= 0) {
	  best_respons_copy[count] = best_respons[i_image];
	  count++;
	}
      }
      assert(count == N_images_included);

      /* Sort the responsabilities and set the active image flag for
	 the worst diffraction patterns to 0. */
      qsort(best_respons_copy, N_images_included, sizeof(real), compare_real);
      real threshold = best_respons_copy[(int)((real)N_images_included*conf.exclude_images_ratio)];
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (active_images[i_image] >= 0) {
	  if (best_respons[i_image]  > threshold) {
	    active_images[i_image] = 1;
	  } else { 
	    active_images[i_image] = 0;
	  }
	}
      }

      /* Repeat the above two steps but for the excluded part. */
      count = 0;
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (active_images[i_image] < 0) {
	  best_respons_copy[count] = best_respons[i_image];
	  count++;
	}
      }
      qsort(best_respons_copy, N_images-N_images_included, sizeof(real), compare_real);
      threshold = best_respons_copy[(int)((real)(N_images-N_images_included)*conf.exclude_images_ratio)];
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (active_images[i_image] < 0) {
	  if (best_respons[i_image] > threshold) {
	    active_images[i_image] = -1;
	  } else {
	    active_images[i_image] = -2;
	  }
	}
      }

      /* Write the list of active images to file. */
      sprintf(filename_buffer, "%s/active_%.4d.h5", conf.output_dir, iteration);
      write_1d_int_array_hdf5(filename_buffer, active_images, N_images);
      free(best_respons_copy);
      free(best_respons);
    }
    /* Aftr the active images list is updated it is copied to the GPU. */
    cuda_copy_int_to_device(active_images, d_active_images, N_images);

    /* Start update scaling second time (test) */
    /*
    if (conf.recover_scaling) {
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	
	cuda_update_scaling_full(d_images, d_slices, d_masks, d_scaling, d_weight_map, N_2d, N_images, slice_start, current_chunk, conf.diff);
      }
    }
    */
    /* End update scaling second time (test) */

    /* Reset the compressed model */
    cuda_reset_model(model,d_model_updated);
    cuda_reset_model(weight,d_weight);

    /* This loop through the slice chunks updates compressed the model. */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }

      /* This function does both recalculate part of the expanded model
	 and compresses this part. The model needs to be divided with
	 the weights outside this loop. */
      
      cuda_update_slices(d_images, d_slices, d_masks,
			 d_respons, d_scaling, d_active_images,
			 N_images, slice_start, current_chunk, N_2d,
			 model,d_model_updated, d_x_coord, d_y_coord,
			 d_z_coord, &d_rotations[slice_start*4],
			 d_weight,images);
    }
    /* cuda_update_slices above needs access to the old and text model
       at the same time. Therefore two models are keept simultaneously.
       The d_model_updated is updated while d_model represents the model
       from last iteration. Afterwords d_model is updated to represent
       the new model. */
    d_model_tmp = d_model_updated;
    d_model_updated = d_model;
    d_model = d_model_tmp;

    /* When all slice chunks have been compressed we need to divide the
       model by the model weights. */
    cuda_divide_model_by_weight(model, d_model, d_weight);

    /* If we are recovering the scaling we need to normalize the model
       to keep scalings from diverging. */
    if (conf.recover_scaling) {
      cuda_normalize_model(model, d_model);
    }

    /* < TEST > */
    /* Blur the model */
    /* if (conf.blur_model) { */
    /*   cuda_blur_model(d_model, conf.model_side, conf.blur_model_sigma); */
    /* } */
    
    /* Copy the new compressed model to the CPU. */
    cuda_copy_model(model, d_model);
    cuda_copy_model(weight, d_weight);

    /* Write the new compressed model to file. */
    for (int i = 0; i < N_model; i++) {
      if (weight->data[i] > 0.0 && model->data[i] > 0.) {
	model_out->mask->data[i] = 1;
	model_out->image->data[i] = sp_cinit(model->data[i],0.0);
      } else {
	model_out->mask->data[i] = 0;
	model_out->image->data[i] = sp_cinit(0., 0.);
      }
    }
    sprintf(filename_buffer,"%s/model_%.4d.h5", conf.output_dir, iteration);
    sp_image_write(model_out,filename_buffer,0);

    /* Write the weights to file. */
    for (int i = 0; i < N_model; i++) {
      model_out->image->data[i] = sp_cinit(weight->data[i], 0.);
      model_out->mask->data[i] = 1;
    }
    sprintf(filename_buffer, "%s/weight_%.4d.h5", conf.output_dir, iteration);
    sp_image_write(model_out, filename_buffer, 0);

    /* Update the state to with a new iteration. This file is
       read by the viewer to keep track of the progress of a
       running analysis. */
    write_state_file_iteration(state_file, iteration);
    end_time = clock();
    printf("iteration took %g s\n", (real) (end_time - start_time) / CLOCKS_PER_SEC);
  }

  /* This is the end of the main loop. After this there will be
     a final compression using individual masks and some cleenup. */

  /* Close files that have been open throughout the aalysis. */
  fclose(likelihood);
  fclose(best_rot_file);
  fclose(fit_file);
  fclose(fit_best_rot_file);
  fclose(radial_fit_file);
  if (conf.calculate_r_free) {
    fclose(r_free);
  }

  /* Reset models for a final compression with individual masks. */
  cuda_reset_model(model,d_model_updated);
  cuda_reset_model(weight,d_weight);

  /* Compress the model one last time for output. This time more
     of the middle data is used bu using individual masks. */
  for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
    if (slice_start + slice_chunk >= N_slices) {
      current_chunk = N_slices - slice_start;
    } else {
      current_chunk = slice_chunk;
    }
    /* This function is different from cuda_update_slices is that
       the individual masks provided as negative values in
       d_images_individual_mask is used instead of d_mask. */
    cuda_update_slices_final(d_images_individual_mask, d_slices, d_masks,
			     d_respons, d_scaling, d_active_images,
			     N_images, slice_start, current_chunk, N_2d,
			     model,d_model_updated, d_x_coord, d_y_coord,
			     d_z_coord, &d_rotations[slice_start*4],
			     d_weight,images);

  }

  /* When all slice chunks have been compressed we need to divide the
     model by the model weights. */
  cuda_divide_model_by_weight(model, d_model_updated, d_weight);

  /* If we are recovering the scaling we need to normalize the model
     to keep scalings from diverging. */
  if (conf.recover_scaling){
    cuda_normalize_model(model, d_model_updated);  
  }

  /* Copy the final result to the CPU */
  cuda_copy_model(model, d_model_updated);

  /* Write the final model to file */
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(model->data[i],0.0);
    if (weight->data[i] > 0.0) {
      model_out->mask->data[i] = 1;
    } else {
      model_out->mask->data[i] = 0;
    }
  }
  /* For this output the spimage values are all given proper values. */
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
  
  sprintf(filename_buffer, "%s/model_final.h5", conf.output_dir);
  sp_image_write(model_out, filename_buffer, 0);

  /* Write the final recovered rotations to file. */
  sprintf(filename_buffer, "%s/final_best_rotations.data", conf.output_dir);
  FILE *final_best_rotations_file = fopen(filename_buffer,"wp");
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
	    rotations[final_best_rotation].q[0], rotations[final_best_rotation].q[1],
	    rotations[final_best_rotation].q[2], rotations[final_best_rotation].q[3]);
  }
  fclose(final_best_rotations_file);
  if (conf.recover_scaling){ 
    close_scaling_file(scaling_dataset, scaling_file);
  }
  close_state_file(state_file);
}
