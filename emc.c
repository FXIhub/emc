//#include "fragmentation.h"
#include <spimage.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include "emc.h"
//#include "rotations.h"
#include <libconfig.h>

int compare_real(real *a, real *b){
  if (*a < *b) {return -1;}
  else if (*a > *b) {return 1;}
  else {return 0;}
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


void get_pixel_from_voxel(real detector_distance, real wavelength,
			  real pixel_size, int x_max, int y_max,
			  const real * voxel, real * pixel){
  const real v_x = voxel[0];
  const real v_y = voxel[1];
  const real v_z = voxel[2];
  real p_x = v_x - 0.5 + x_max/2;
  real p_y = v_y - 0.5 + y_max/2;
  real pixel_r = sqrt(v_x*v_x + v_y*v_y);
  real real_r = pixel_r*pixel_size;
  real angle_r = atan2(real_r,detector_distance);
  real fourier_r = sinf(angle_r)/wavelength;
  real fourier_z = (1.0f - cosf(angle_r))/wavelength;
  real calc_z = fourier_z/fourier_r*pixel_r;
  if(fabs(calc_z - v_z) <= 0.5){
    pixel[0] = p_x;
    pixel[1] = p_y;    
  }else{
    pixel[0] = -1;
  }

}

void test_get_pixel_from_voxel(int side, real pixel_size, real detector_distance, real wavelength,
			       sp_matrix *x_coordinates, sp_matrix *y_coordinates, sp_matrix *z_coordinates) {
  const int x_max = side;
  const int y_max = side;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      real voxel[3];
      voxel[0] = rint(sp_matrix_get(x_coordinates,x,y));
      voxel[1] = rint(sp_matrix_get(y_coordinates,x,y));
      voxel[2] = rint(sp_matrix_get(z_coordinates,x,y));
      real pixel[2];
      get_pixel_from_voxel(detector_distance, wavelength,
			   pixel_size, x_max, y_max,
			   voxel, pixel);
      if(lrint(pixel[0]) != x){
	printf("x - %f and %d don't match\n",pixel[0],x);
      }
      if(lrint(pixel[1]) != y){
	printf("y - %f and %d don't match\n",pixel[1],y);
      }      
    }
  }
}

void get_slice(sp_3matrix *model, sp_matrix *slice, Quaternion *rot,
	       sp_matrix *x_coordinates, sp_matrix *y_coordinates,
	       sp_matrix *z_coordinates)
{
  const int x_max = sp_matrix_rows(slice);
  const int y_max = sp_matrix_cols(slice);
  int pixel_r;
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      /* This is just a matrix multiplication with rot */
      new_x =
	(rot->q[0]*rot->q[0] + rot->q[1]*rot->q[1] -
	 rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +
	(2.0*rot->q[1]*rot->q[2] -
	 2.0*rot->q[0]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +
	(2.0*rot->q[1]*rot->q[3] +
	 2.0*rot->q[0]*rot->q[2])*sp_matrix_get(z_coordinates,x,y);
      new_y =
	(2.0*rot->q[1]*rot->q[2] +
	 2.0*rot->q[0]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +
	(rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] +
	 rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +
	(2.0*rot->q[2]*rot->q[3] -
	 2.0*rot->q[0]*rot->q[1])*sp_matrix_get(z_coordinates,x,y);
      new_z =
	(2.0*rot->q[1]*rot->q[3] -
	 2.0*rot->q[0]*rot->q[2])*sp_matrix_get(x_coordinates,x,y) +
	(2.0*rot->q[2]*rot->q[3] +
	 2.0*rot->q[0]*rot->q[1])*sp_matrix_get(y_coordinates,x,y) +
	(rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] -
	 rot->q[2]*rot->q[2] + rot->q[3]*rot->q[3])*sp_matrix_get(z_coordinates,x,y);
      round_x = round((real)sp_3matrix_x(model)/2.0 + 0.5 + new_x);
      round_y = round((real)sp_3matrix_y(model)/2.0 + 0.5 + new_y);
      round_z = round((real)sp_3matrix_z(model)/2.0 + 0.5 + new_z);
      if (round_x > 0 && round_x < sp_3matrix_x(model) &&
	  round_y > 0 && round_y < sp_3matrix_y(model) &&
	  round_z > 0 && round_z < sp_3matrix_z(model)) {
	sp_matrix_set(slice,x,y,sp_3matrix_get(model,round_x,round_y,round_z));
      } else {
	sp_matrix_set(slice,x,y,0.0);
      }
    }
  }
}

void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
		  sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
		  sp_matrix *y_coordinates, sp_matrix *z_coordinates)
{
  const int x_max = sp_matrix_rows(slice);
  const int y_max = sp_matrix_cols(slice);
  int pixel_r;
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
	round_x = round((real)sp_3matrix_x(model)/2.0 + 0.5 + new_x);
	round_y = round((real)sp_3matrix_y(model)/2.0 + 0.5 + new_y);
	round_z = round((real)sp_3matrix_z(model)/2.0 + 0.5 + new_z);
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

real update_slice(sp_matrix ** images, sp_matrix ** slices, sp_imatrix * mask,
		  real * respons, real * scaling,
		  int N_images, int N_2d, int i_slice){
  real total_respons = 0.0;
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] != 0) {
	slices[i_slice]->data[i] += images[i_image]->data[i]*
	  respons[i_slice*N_images+i_image]/scaling[i_image];
      }
    }
    total_respons += respons[i_slice*N_images+i_image];
  }
  return total_respons;
}

real slice_weighting(sp_matrix ** images, sp_matrix ** slices,sp_imatrix * mask,
		     real * respons, real * scaling,
		     int N_slices, int N_2d, int i_image, int N_images){
  real weighted_power = 0;
  for (int i_slice = 0; i_slice < N_slices; i_slice++) { 
    real correlation = 0.0;
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] != 0) {
	correlation += images[i_image]->data[i]*slices[i_slice]->data[i];
      }
    }
    weighted_power += respons[i_slice*N_images+i_image]*correlation;
  }  
  return weighted_power;
}

void calculate_normalization(sp_matrix **images, int size, sp_matrix *average, sp_matrix *scale)
{
  /* create radius matrix */
  sp_matrix *radius = sp_matrix_alloc(sp_matrix_rows(images[0]),sp_matrix_cols(images[0]));
  const int x_max = sp_matrix_rows(radius);
  const int y_max = sp_matrix_cols(radius);
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      sp_matrix_set(radius,x,y,sqrt(pow((real)x-(real)x_max/2.0+0.5,2) +
				    pow((real)y-(real)y_max/2.0+0.5,2)));
    }
  }

  /* calculate average */
  const int length = ceil(sqrt(pow(x_max,2)+pow(y_max,2))/2.0);
  real histogram[length];
  int count[length];
  for (int i = 0; i < length; i++) {
    histogram[i] = 0.0;
    count[i] = 0;
  }
  for (int i_image = 0; i_image < size; i_image++) {
    for (int x = 0; x < x_max; x++) {
      for (int y = 0; y < y_max; y++) {
	histogram[(int)floor(sp_matrix_get(radius,x,y))] += sp_matrix_get(images[i_image],x,y);
	count[(int)floor(sp_matrix_get(radius,x,y))] += 1;
      }
    }
  }
  printf("length = %d\n",length);
  for (int i = 0; i < length; i++) {
    if (count[i] > 0) {
      histogram[i] /= count[i];
    }
  }
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      sp_matrix_set(average,x,y,histogram[(int)floor(sp_matrix_get(radius,x,y))]);
    }
  }


  /* calculate scale */
  for (int i = 0; i < length; i++) {
    histogram[i] = 0.0;
    count[i] = 0;
  }
  for (int i_image = 0; i_image < size; i_image++) {
    for (int x = 0; x < x_max; x++) {
      for (int y = 0; y < y_max; y++) {
	histogram[(int)floor(sp_matrix_get(radius,x,y))] += pow(sp_matrix_get(images[i_image],x,y) -
							   sp_matrix_get(average,x,y),2);
	count[(int)floor(sp_matrix_get(radius,x,y))] += 1;
      }
    }
  }
  for (int i = 0; i < length; i++) {
    histogram[i] /= count[i];
  }
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      if (sp_matrix_get(radius,x,y) < sp_min(x_max,y_max)/2.0) {
	sp_matrix_set(scale,x,y,1.0/sqrt(histogram[(int)floor(sp_matrix_get(radius,x,y))]));
      } else {
	sp_matrix_set(scale,x,y,0.0);
      }
    }
  }
  /*
  Image *out = sp_image_alloc(x_max,y_max,1);
  for (int i = 0; i < x_max*y_max; i++) {
    out->image->data[i] = sp_cinit(average->data[i],0.0);
  }
  sp_image_write(out,"debug_average.h5",0);
  for (int i = 0; i < x_max*y_max; i++) {
    out->image->data[i] = sp_cinit(scale->data[i],0.0);
  }
  sp_image_write(out,"debug_scale.h5",0);
  exit(1);
  */
}

real calculate_correlation(sp_matrix *slice, sp_matrix *image,
			   sp_matrix *average, sp_matrix *scale)
{
  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  for (int i = 0; i < i_max; i++) {
    sum += (slice->data[i] - average->data[i]) *
      (image->data[i] - average->data[i]) * pow(scale->data[i],2);
  }
  return sum;
}

/* This responsability does not yet take scaling of patterns into accoutnt. */
real calculate_responsability_absolute(sp_matrix *slice, sp_matrix *image, sp_imatrix *mask, real sigma, real scaling)
{

  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  int count = 0;
  for (int i = 0; i < i_max; i++) {
    if (mask->data[i] != 0) {
      sum += pow(slice->data[i] - image->data[i]/scaling,2);
      count++;
    }
  }
  //return exp(-sum/2.0/(real)count/pow(sigma,2));
  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

real calculate_responsability_poisson(sp_matrix *slice, sp_matrix *image, sp_imatrix *mask, real sigma, real scaling)
{
  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  int count = 0;
  for (int i = 0; i < i_max; i++) {
    if (mask->data[i] != 0) {
      sum += pow(slice->data[i] - image->data[i]/scaling,2) / (1.0+image->data[i]) * scaling;
      count++;
    }
  }
  return exp(-sum/2.0/(real)count/pow(sigma,2));
}

real calculate_responsability_relative(sp_matrix *slice, sp_matrix *image, sp_imatrix *mask, real sigma, real scaling)
{
  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  int count = 0;
  for (int i = 0; i < i_max; i++) {
    if (mask->data[i] != 0) {
      sum += pow((slice->data[i] - image->data[i]/scaling + sqrt(image->data[i]+1.0)/scaling) / (slice->data[i] + (1.0+image->data[i]/scaling)),2);
      count++;
    }
  }
  return exp(-sum/2.0/(real)count/pow(sigma,2));
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
  config_lookup_float(&config,"sigma",&config_out.sigma);
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
  config_lookup_string(&config,"model_file",&config_out.model_file);
  config_lookup_bool(&config,"exclude_images",&config_out.exclude_images);
  config_lookup_float(&config,"exclude_ratio",&config_out.exclude_ratio);

  return config_out;
}

sp_matrix **read_images(Configuration conf, sp_imatrix **masks)
{
  sp_matrix **images = malloc(conf.N_images*sizeof(sp_matrix *));
  //masks = malloc(conf.N_images*sizeof(sp_imatrix *));
  Image *img;
  real new_intensity;
  real *intensities = malloc(conf.N_images*sizeof(real));
  real scale_sum = 0.0;
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
    /*
    for (int x = 0; x < conf.model_side; x++) {
      for (int y = 0; y < conf.model_side; y++) {

	sp_matrix_set(images[i],x,y,
		      sp_cabs(sp_image_get(img,
					   (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+
						 sp_image_x(img)/2-0.5),
					   (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+
						 sp_image_y(img)/2-0.5),0)));
	if (sp_image_mask_get(img,
			      (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+
				    sp_image_x(img)/2-0.5),
			      (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+
				    sp_image_y(img)/2-0.5),0)) {
	  sp_imatrix_set(masks[i],x,y,1);
	} else {
	  sp_imatrix_set(masks[i],x,y,0);
	}
      }
    }
    */

    real pixel_sum;
    int mask_sum;
    for (int x = 0; x < conf.model_side; x++) {
      for (int y = 0; y < conf.model_side; y++) {
	pixel_sum = 0.0;
	mask_sum = 0;
	for (int xb = 0; xb < conf.read_stride; xb++) {
	  for (int yb = 0; yb < conf.read_stride; yb++) {
	    pixel_sum += sp_cabs(sp_image_get(img,(int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,(int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0));
	    mask_sum += sp_image_mask_get(img,(int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+sp_image_x(img)/2-0.5)+xb,(int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+sp_image_y(img)/2-0.5)+yb,0);
	  }
	}
	sp_matrix_set(images[i],x,y,pixel_sum);
	if (mask_sum == 3) {
	  sp_imatrix_set(masks[i],x,y,1);
	} else {
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
  real sum;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    sum = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] == 1) {
	sum += images[i_image]->data[i];
      }
    }
    sum = (real)N_2d / sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}

/* normalize images so average pixel value is 1.0 */
void normalize_images_individual_mask(sp_matrix **images, sp_imatrix **masks,
				      Configuration conf)
{
  real sum;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    sum = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 1) {
	sum += images[i_image]->data[i];
      }
    }
    sum = (real)N_2d / sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}


int main(int argc, char **argv)
{
  Configuration conf;
  if (argc > 1) {
    conf = read_configuration_file(argv[1]);
  } else {
    conf = read_configuration_file("emc.conf");
  }
  const int start_iteration = 0;
  const int rescale_intensity = 0;
  const real intensity_fluct = 0.2; //when reading images they are randomly rescaled. Temporary.

  const int N_images = conf.N_images;
  const int slice_chunk = conf.slice_chunk;
  const int output_period = 10;
  const int n = conf.rotations_n;
  const int N_2d = conf.model_side*conf.model_side;
  char buffer[1000];

  Quaternion **rotations;
  real *weights;
  const long long int N_slices = generate_rotation_list(n,&rotations,&weights);
  printf("%lld rotations sampled\n",N_slices);

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
  mkdir("debug",0777);
  mkdir("output",0777);

  if (conf.normalize_images) {
    //normalize_images(images, mask, conf);
    normalize_images_individual_mask(images, masks, conf);
  }
  /* output images after preprocessing */
  Image *write_image = sp_image_alloc(conf.model_side,conf.model_side,1);
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i]) {
	sp_real(write_image->image->data[i]) = images[i_image]->data[i];
      } else {
	sp_real(write_image->image->data[i]) = 0.0;
      }
      write_image->mask->data[i] = masks[i_image]->data[i];
    }
    sprintf(buffer, "debug/image_%.4d.png", i_image);
    sp_image_write(write_image, buffer, SpColormapJet|SpColormapLogScale);
    sprintf(buffer, "debug/image_%.4d.h5", i_image);
    sp_image_write(write_image, buffer, 0);
  }
  sp_image_free(write_image);

  sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength,
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
  real model_d = 1.0/(conf.pixel_size*(real)conf.detector_size/conf.detector_distance*
		      conf.wavelength);
  sp_3matrix *weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  const long long int N_model = conf.model_side*conf.model_side*conf.model_side;

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
	  r = (int)sqrt(pow((real)x - conf.model_side/2.0 + 0.5,2) +
			pow((real)y - conf.model_side/2.0 + 0.5,2));
	  if (r < conf.model_side/2.0) {
	    radavg[r] += sp_matrix_get(images[i_image],x,y);
	    radavg_count[r] += 1;
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
					radavg[r+1]*(rad - (real)r)));
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
  }

  real *scaling = malloc(N_images*sizeof(real));
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
  real sum, total_respons, overal_respons, min_resp, max_resp;
  real image_power, weighted_power, correlation, scaling_error;
  real model_sum;
  FILE *likelihood = fopen("likelihood.data","wp");

  real * slices;
  cuda_allocate_slices(&slices,conf.model_side,slice_chunk); //was N_slices before
  //real * slices_on_host = malloc(N_slices*N_2d*sizeof(real));
  real * d_model;
  cuda_allocate_model(&d_model,model);
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
  real * d_scaling;
  cuda_allocate_scaling(&d_scaling,N_images);
  real *d_weighted_power;
  cuda_allocate_real(&d_weighted_power,N_images);
  real *fit = malloc(N_images*sizeof(real));
  real *d_fit;
  cuda_allocate_real(&d_fit,N_images);
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
  
  real *sorted_resp = malloc(N_slices*sizeof(real));
  real *total_sorted_resp = malloc(N_slices*sizeof(real));

  FILE *fit_file = fopen("output/fit.data","wp");
  FILE *scaling_file = fopen("output/scaling.data","wp");
  FILE *radial_fit_file = fopen("output/radial_fit.data","wp");
  FILE *responsabilities_file;// = fopen("output/responsability.data","wp");
  FILE *sorted_resp_file = fopen("output/total_resp.data","wp");

  int current_chunk;
  for (int iteration = start_iteration; iteration < conf.max_iterations; iteration++) {
    sum = cuda_model_max(d_model,N_model);
    printf("model max = %g\n",sum);

    printf("iteration %d\n", iteration);
    /* get slices */
    clock_t t_i = clock();
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      if ((slice_start/slice_chunk)%output_period == 0) {
	printf("calculate presponsabilities chunk %d\n", slice_start/slice_chunk);
      }

      cuda_get_slices(model,d_model,slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);
      //cuda_copy_slice_chunk_to_host(slices_on_host, slices, slice_start, current_chunk, N_2d);


      /* calculate responsabilities */
    
      cuda_calculate_responsabilities(slices, d_images, d_mask,
				      conf.sigma, d_scaling,d_respons, 
				      N_2d, N_images, slice_start,
				      current_chunk);

    }
    printf("calculated responsabilities\n");
    clock_t t_e = clock();
    printf("Expansion time = %fs\n",(real)(t_e - t_i)/(real)CLOCKS_PER_SEC);

    cuda_calculate_responsabilities_sum(respons, d_respons, N_slices,
					N_images);
    
    printf("normalize resp\n");
    cuda_normalize_responsabilities(d_respons, N_slices, N_images);
    cuda_copy_real_to_host(respons, d_respons, N_slices*N_images);
    printf("normalize resp done\n");

    /* plot occupancy plot */
    int N_r = 30;
    int N_a = 2*N_r;
    /*
    int N_r = 80;
    int N_a = 60;
    */
    real a,b,c;
    int a_int, r_int;
    Image *rotation_map = sp_image_alloc(N_r,N_a,1);
    for (int i_image = 0; i_image < 10; i_image++) {
      for (int i = 0; i < sp_image_size(rotation_map); i++) {
	sp_real(rotation_map->image->data[i]) = 0.0;
	sp_imag(rotation_map->image->data[i]) = 0.0;
	rotation_map->mask->data[i] = 0;
      }
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	quaternion_to_euler(rotations[i_slice],&a,&b,&c);
	//r_int = ((int)((b+2*N_r)/M_PI*(real)N_r))%N_r;
	r_int = ((int)((b+M_PI/2.0)/M_PI*(real)(N_r-1)))%N_r;
	a_int = ((int)((a+M_PI)/2.0/M_PI*(real)N_a))%N_a;
	if (r_int < 0 || r_int >= N_r || a_int < 0 || a_int >= N_a) {
	  printf("r_int = %d, a_int = %d\n",r_int,a_int);
	}
	sp_image_set(rotation_map,r_int,a_int,0,sp_cinit(sp_real(sp_image_get(rotation_map,r_int,a_int,0))+respons[i_slice*N_images+i_image],0.0));
	sp_image_mask_set(rotation_map,r_int,a_int,0,sp_image_mask_get(rotation_map,r_int,a_int,0)+1);
      }
      for (int i = 0; i < sp_image_size(rotation_map); i++) {
	if (rotation_map->mask->data[i] > 0) {
	  sp_real(rotation_map->image->data[i]) /= (real) rotation_map->mask->data[i];
	} else {
	  sp_real(rotation_map->image->data[i]) = 0.0;
	}
	sp_imag(rotation_map->image->data[i]) = 0.0;
      }

      sprintf(buffer,"debug/rotation_map_%.4d_%.4d.h5",i_image,iteration);
      sp_image_write(rotation_map,buffer,0);
      sprintf(buffer,"debug/rotation_map_%.4d_%.4d.png",i_image,iteration);
      sp_image_write(rotation_map,buffer,SpColormapJet);
    }
    for (int i = 0; i < sp_image_size(rotation_map); i++) {
      sp_real(rotation_map->image->data[i]) = 0.0;
      sp_imag(rotation_map->image->data[i]) = 0.0;
      rotation_map->mask->data[i] = 0;
    }
    for (int i_image = 0; i_image < N_images; i_image++) {
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	quaternion_to_euler(rotations[i_slice],&a,&b,&c);
	//r_int = ((int)((b+2*N_r)/M_PI*(real)N_r))%N_r;
	r_int = ((int)((b+M_PI/2.0)/M_PI*(real)(N_r-1)))%N_r;
	a_int = ((int)((a+M_PI)/2.0/M_PI*(real)N_a))%N_a;
	if (r_int < 0 || r_int >= N_r || a_int < 0 || a_int >= N_a) {
	  printf("r_int = %d, a_int = %d\n",r_int,a_int);
	}
	sp_image_set(rotation_map,r_int,a_int,0,sp_cinit(sp_real(sp_image_get(rotation_map,r_int,a_int,0))+respons[i_slice*N_images+i_image],0.0));
	sp_image_mask_set(rotation_map,r_int,a_int,0,sp_image_mask_get(rotation_map,r_int,a_int,0)+1);
      }
    }
    for (int i = 0; i < sp_image_size(rotation_map); i++) {
      if (rotation_map->mask->data[i] > 0) {
	sp_real(rotation_map->image->data[i]) /= (real) rotation_map->mask->data[i];
      } else {
	sp_real(rotation_map->image->data[i]) = 0.0;
      }
      sp_imag(rotation_map->image->data[i]) = 0.0;
    }
    sprintf(buffer,"debug/rotation_map_total_%.4d.h5",iteration);
    sp_image_write(rotation_map,buffer,0);
    sprintf(buffer,"debug/rotation_map_total_%.4d.png",iteration);
    sp_image_write(rotation_map,buffer,SpColormapJet);
      
    sp_image_free(rotation_map);

    /* output responsabilities */
    sprintf(buffer,"output/responsabilities_%.4d.data",iteration);
    responsabilities_file = fopen(buffer, "wp");
    for (int i_image = 0; i_image < 5; i_image++) {
      for(int i_slice = 0; i_slice < N_slices; i_slice++) {
	fprintf(responsabilities_file, "%g ", respons[i_slice*N_images+i_image]);
      }
      fprintf(responsabilities_file, "\n");
    }
    fclose(responsabilities_file);

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
      fprintf(sorted_resp_file,"%g ", total_sorted_resp[i_slice] / (real) N_images);
    }
    fprintf(sorted_resp_file,"\n");
    fflush(sorted_resp_file);
	

    /* check how well every image fit */
    int radial_fit_n = 100000;
    cuda_set_to_zero(d_fit,N_images);
    cuda_set_to_zero(d_radial_fit,conf.model_side/2);
    cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2);
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      cuda_get_slices(model,d_model,slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);
      
      cuda_calculate_fit(slices, d_images, d_mask, d_scaling,
			 d_respons, d_fit, conf.sigma, N_2d, N_images,
			 slice_start, current_chunk);
      if (iteration % radial_fit_n == 0 && iteration != 0 || iteration == conf.max_iterations-1) {
	cuda_calculate_radial_fit(slices, d_images, d_mask,
				  d_scaling, d_respons, d_radial_fit,
				  d_radial_fit_weight, d_radius,
				  N_2d, conf.model_side, N_images, slice_start,
				  slice_chunk);
      }
    }

    cuda_copy_real_to_host(fit, d_fit, N_images);
    for (int i_image = 0; i_image < N_images; i_image++) {
      fprintf(fit_file, "%g ", fit[i_image]);
    }
    fprintf(fit_file, "\n");
    fflush(fit_file);

    /* finis calculating radial fit */
    if ((iteration % radial_fit_n == 0 && iteration != 0) || iteration == conf.max_iterations-1) {
      cuda_copy_real_to_host(radial_fit, d_radial_fit, conf.model_side/2);
      cuda_copy_real_to_host(radial_fit_weight, d_radial_fit_weight, conf.model_side/2);
      for (int i = 0; i < conf.model_side/2; i++) {
	printf("%d: %g %g\n", i, radial_fit[i], radial_fit_weight[i]);
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
    /* calculate likelihood */
    
    t_i = clock();

    total_respons = cuda_total_respons(d_respons,respons,N_images*N_slices);
    printf("calculated total resp\n");

    fprintf(likelihood,"%g\n",total_respons);
    printf("likelihood = %g\n",total_respons);
    fflush(likelihood);
  
    if (conf.known_intensity == 0) {
      /* update scaling */
      clock_t local_t_i = clock();
      cuda_set_to_zero(d_weighted_power,N_images);
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	if ((slice_start/slice_chunk)%output_period == 0) {
	  printf("calculate weighted_power chunk %d\n", slice_start/slice_chunk);
	}

	cuda_get_slices(model, d_model, slices, d_rotations,
			d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	//cuda_copy_slice_chunk_to_device(slices_on_host, slices, slice_start, slice_chunk, N_2d);

	cuda_update_weighted_power(d_images, slices, d_mask,
				   d_respons, d_weighted_power, N_images,
				   slice_start, current_chunk, N_2d);


      }
      cuda_update_scaling(d_images, d_mask, d_scaling, d_weighted_power,
			  N_images, N_slices, N_2d, scaling);
      clock_t local_t_e = clock();
      printf("Update scaling time = %fs\n",(real)(local_t_e - local_t_i)/(real)CLOCKS_PER_SEC);

      /* normalize scaling */
      /*
      real scaling_sum = 0.0;
      for (int i_image = 0; i_image < N_images; i_image++) {
	//scaling_sum += scaling[i_image];
	scaling_sum += 1.0/scaling[i_image];
      }
      //scaling_sum = (real)N_images / scaling_sum;
      scaling_sum /= (real)N_images;
      for (int i_image = 0; i_image < N_images; i_image++) {
	scaling[i_image] *= scaling_sum;
      }
      cuda_copy_real_to_device(scaling, d_scaling, N_images);
      */

      if (iteration%1 == 0) {
	real average_scaling = 0.0;
	for (int i_image = 0; i_image < N_images; i_image++) {
	  average_scaling += scaling[i_image];
	}
	average_scaling /= (real)N_images;
	//printf("scaling:\n");
	real variance = 0.0;
	for (int i_image = 0; i_image < N_images; i_image++) {
	  //printf("%g, ", scaling[i_image] / average_scaling);
	  variance += pow(scaling[i_image] /average_scaling - 1.0, 2);
	}
	//printf("\n");

	variance /= (real)N_images;
	printf("scaling std = %g\n", sqrt(variance));
      }

      for (int i_image = 0; i_image < N_images; i_image++) {
	fprintf(scaling_file, "%g ", scaling[i_image]);
      }
    }
    fprintf(scaling_file, "\n");
    fflush(scaling_file);

    /* reset model */    
    cuda_reset_model(model,d_model_updated);
    cuda_reset_model(weight,d_weight);
    printf("models reset\n");



    int my_array[261] = {0, 1, 0, 1, 1, 0, 1, 1, 1, 1, //   0
			 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, //  10
			 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, //  20
			 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, //  30
			 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, //  40
			 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, //  50
			 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, //  60
			 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, //  70
			 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, //  80
			 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, //  90
			 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, // 100
			 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, // 110
			 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, // 120
			 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, // 130
			 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, // 140
			 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, // 150
			 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, // 160
			 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // 170
			 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, // 180
			 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, // 190
			 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, // 200
			 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 210 
			 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 220
			 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 230
			 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 240
			 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // 250
			 1};                           // 260
    if (iteration == 0) {
      /*
      for (int i_image = 0; i_image < N_images; i_image++) {
	active_images[i_image] = my_array[i_image];
      }
      */

      for (int i_image = 0; i_image < N_images; i_image++) {
	active_images[i_image] = 1;
      }

    } else if (iteration == 100000) {
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (log(fit[i_image]) < -10) {
	  active_images[i_image] = 1;
	} else {
	  active_images[i_image] = 0;
	}
      }
    }
    if (conf.exclude_images == 1 && iteration > -1) {
      real *fit_copy = malloc(N_images*sizeof(real));
      memcpy(fit_copy,fit,N_images*sizeof(real));
      qsort(fit_copy, N_images, sizeof(real), compare_real);
      real threshold = fit_copy[(int)((real)N_images*conf.exclude_ratio)];
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (i_image != 0 && i_image % 10 == 0) printf(" ");
	if (fit[i_image] > threshold && my_array[i_image] > 0) {
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

    /* update slices */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      if ((slice_start/slice_chunk)%output_period == 0) {
	printf("update slices chunk %d\n", slice_start/slice_chunk);
      }

      cuda_get_slices(model, d_model, slices, d_rotations,
		      d_x_coord, d_y_coord, d_z_coord,
		      slice_start, current_chunk);
      //cuda_copy_slice_chunk_to_device(slices_on_host, slices, slice_start, slice_chunk, N_2d);

      cuda_update_slices(d_images, slices, d_mask,
			 d_respons, d_scaling, d_active_images, N_images, slice_start, current_chunk, N_2d,
			 model,d_model_updated, d_x_coord, d_y_coord,
			 d_z_coord, &d_rotations[slice_start*4], &weights[slice_start], d_weight,images);
    }
    d_model_tmp = d_model_updated;
    d_model_updated = d_model;
    d_model = d_model_tmp;

    cuda_copy_model(model, d_model);

    printf("updated slices\n");
    t_e = clock();
    printf("Maximize time = %fs\n",(real)(t_e - t_i)/(real)CLOCKS_PER_SEC);

    t_i = clock();
    cuda_normalize_model(model, d_model,d_weight);
    printf("compressed\n");
    t_e = clock();
    printf("Compression time = %fms\n",1000.0*(t_e - t_i)/CLOCKS_PER_SEC);

    /* write output */
    for (int i = 0; i < N_model; i++) {
      model_out->image->data[i] = sp_cinit(model->data[i],0.0);
      if (weight->data[i] > 0.0) {
	model_out->mask->data[i] = 1;
      } else {
	model_out->mask->data[i] = 0;
      }
    }

    sprintf(buffer,"output/model_%.4d.h5",iteration);
    sp_image_write(model_out,buffer,0);
    printf("wrote model\n");
  }
  fclose(likelihood);
  fclose(fit_file);
  fclose(scaling_file);
  fclose(radial_fit_file);
  fclose(sorted_resp_file);

  cuda_reset_model(model,d_model_updated);
  cuda_reset_model(weight,d_weight);

  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (masks[i_image]->data[i] == 0) {
	images[i_image]->data[i] = -1.0;
      }
    }
  }

  real * d_masked_images;
  cuda_allocate_images(&d_masked_images,images,N_images);

  real *h_slices = malloc(slice_chunk*N_2d*sizeof(real));

  /* put together the model one last time for output.
     This time more of the middle data is used */
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

    cuda_update_slices_final(d_masked_images, slices, h_slices, d_mask,
			     d_respons, d_scaling, d_active_images,
			     N_images, slice_start, current_chunk, N_2d,
			     model,d_model_updated, d_x_coord, d_y_coord,
			     d_z_coord, &d_rotations[slice_start*4],
			     &weights[slice_start], d_weight,images);
    /*
    write_image = sp_image_alloc(conf.model_side,conf.model_side,1);
    for (int i_slice = 0; i_slice < current_chunk; i_slice++) {
      for (int i = 0; i < N_2d; i++) {
	sp_real(write_image->image->data[i]) = h_slices[i_slice*N_2d + i];
	sp_imag(write_image->image->data[i]) = 0.0;
      }
      sprintf(buffer,"debug/slice_%.6d.png",slice_start+i_slice);
      sp_image_write(write_image,buffer,SpColormapLogScale|SpColormapJet);
    }
    sp_image_free(write_image);
    */
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

  cuda_normalize_model(model, d_model_updated,d_weight);  
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

  sp_image_write(model_out,"output/model_final.h5",0);

  
  FILE *final_best_rotations = fopen("debug/final_best_rotations.data","wp");
  real highest_resp, this_resp;
  int best_rotation;
  for (int i_image = 0; i_image < N_images; i_image++) {
    best_rotation = 0;
    highest_resp = respons[0*N_images+i_image];
    for (int i_slice = 1; i_slice < N_slices; i_slice++) {
      this_resp = respons[i_slice*N_images+i_image];
      if (this_resp > highest_resp) {
	best_rotation = i_slice;
	highest_resp = this_resp;
      }
    }
    fprintf(final_best_rotations, "%g %g %g %g\n",
	    rotations[best_rotation]->q[0], rotations[best_rotation]->q[1],
	    rotations[best_rotation]->q[2], rotations[best_rotation]->q[3]);
  }
  fclose(final_best_rotations);
}
