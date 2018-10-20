/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

/* Only used to provide to qsort. */
#include "emc_math.h"

int compare_real(const void *pa, const void *pb) {
    real a = *(const real*)pa;
    real b = *(const real*)pb;
    
    if (a < b) {return -1;}
    else if (a > b) {return 1;}
    else {return 0;}
}


/* Precalculate coordinates. These coordinates represent an
 Ewald sphere with the xy plane with liftoff in the z direction.
 These coordinates then only have to be rotated to get coordinates
 for expansion and compression. */
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
                 rot.q[2]*rot.q[2] - rot.q[3]*rot.q[3])*sp_matrix_get(z_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
                (2.0*rot.q[1]*rot.q[2] -
                 2.0*rot.q[0]*rot.q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
                (2.0*rot.q[1]*rot.q[3] +
                 2.0*rot.q[0]*rot.q[2])*sp_matrix_get(x_coordinates,x,y);
                new_y =
                (2.0*rot.q[1]*rot.q[2] +
                 2.0*rot.q[0]*rot.q[3])*sp_matrix_get(z_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
                (rot.q[0]*rot.q[0] - rot.q[1]*rot.q[1] +
                 rot.q[2]*rot.q[2] - rot.q[3]*rot.q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
                (2.0*rot.q[2]*rot.q[3] -
                 2.0*rot.q[0]*rot.q[1])*sp_matrix_get(x_coordinates,x,y);
                new_z =
                (2.0*rot.q[1]*rot.q[3] -
                 2.0*rot.q[0]*rot.q[2])*sp_matrix_get(z_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
                (2.0*rot.q[2]*rot.q[3] +
                 2.0*rot.q[0]*rot.q[1])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
                (rot.q[0]*rot.q[0] - rot.q[1]*rot.q[1] -
                 rot.q[2]*rot.q[2] + rot.q[3]*rot.q[3])*sp_matrix_get(x_coordinates,x,y);
                
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
                    //sp_3matrix_set(weight,round_x,round_y,round_z,sp_3matrix_get(weight,round_x,round_y,round_z)+w);                    
                    sp_3matrix_set(weight,round_x,round_y,round_z,sp_3matrix_get(weight,round_x,round_y,round_z)+w);
                }
            }// end of if
        }
    }
}



/* Normalize all diffraction patterns so that the average pixel
 value is 1.0 in each pattern. Use the common mask for the
 normalization. */
void normalize_images(sp_matrix **images, sp_imatrix *mask, Configuration conf, real central_part_radius = 10)
{
    /*real sum, count;
    int N_2d = conf.model_side*conf.model_side;
    for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
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
    }*/
    if (conf.normalize_images) {
        if (!conf.recover_scaling) {
            normalize_images_preserve_scaling(images, mask, conf);
        } else {
            normalize_images_central_part(images, mask, central_part_radius, conf);
        }
    }
}

/* Normalize all diffraction patterns so that the average pixel value in
 each patterns in a circle of the specified radius is 0. An input radius
 of 0 means that the full image is used. */
void normalize_images_central_part(sp_matrix ** const images, const sp_imatrix * const mask, real radius, const Configuration conf) {
    const int x_max = conf.model_side;
    const int y_max = conf.model_side;
    /* If the radius is 0 we use the full image by setting the
     radius to a large number. */
    if (radius == 0) {
        radius = sqrt(pow(x_max, 2) + pow(y_max, 2))/2. + 2;
    }
    
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
void normalize_images_preserve_scaling(sp_matrix ** images, sp_imatrix *mask, Configuration conf) {
    int N_2d = conf.model_side*conf.model_side;
    real sum = 0.;
    real count = 0.;
    for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i] == 1) {
                sum += images[i_image]->data[i];
                count += 1.;
            }
        }
    }
    sum = (count*(real)conf.number_of_images) / sum;
    for (int i_image = 0; i_image < conf.number_of_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            images[i_image]->data[i] *= sum;
        }
    }
}



/* Assemble the model from the given diffraction patterns
 with a random orientation assigned to each. */
void create_initial_model_random_orientations(sp_3matrix *model, sp_3matrix *weight, sp_matrix **images,
                                                     const int N_images, sp_imatrix *mask, sp_matrix *x_coordinates,
                                                     sp_matrix *y_coordinates, sp_matrix *z_coordinates, gsl_rng *rng) {
    const int N_model = sp_3matrix_size(model);
    Quaternion random_rot;
    for (int i = 0; i < N_images; i++) {
        random_rot = quaternion_random(rng);
        insert_slice(model, weight, images[i], mask, 1.0, random_rot,
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
void create_initial_model_given_orientations(sp_3matrix *model, sp_3matrix *weight, sp_matrix **images, const int N_images,
                                                    sp_imatrix *mask, sp_matrix *x_coordinates, sp_matrix *y_coordinates,
                                                    sp_matrix *z_coordinates, const char *init_rotations_file) {
    const int N_model = sp_3matrix_size(model);
    FILE *given_rotations_file = fopen(init_rotations_file, "r");
        //Quaternion *this_rotation = quaternion_alloc();
    Quaternion this_rotation;
    for (int i_image = 0; i_image < N_images; i_image++) {
        fscanf(given_rotations_file, "%g %g %g %g\n", &(this_rotation.q[0]), &(this_rotation.q[1]), &(this_rotation.q[2]), &(this_rotation.q[3]));
        insert_slice(model, weight, images[i_image], mask, 1., this_rotation,
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
void create_initial_model_file(sp_3matrix *model, const char *model_file) {
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


int get_allocate_len(int ntasks, int N_slices, int taskid){
    int slice_start = taskid* N_slices/ntasks;
    int slice_end =  (taskid+1)* N_slices/ntasks;
    if (taskid == ntasks -1)
        slice_end = N_slices;
    return slice_end - slice_start;
}
void copy_real(int len, real* source , real* dst){
    memcpy(dst,source,len *sizeof(real));
}


//sum up ori and tmp to ori
void sum_vectors(real* ori, real* tmp, int len){
    for(int i = 0; i< len; i++)
        ori[i] += tmp[i];
}
//take the log of vector a
void log_vector(real* a, int len ){
    for(int i = 0; i< len; i++)
        a[i]= log(a[i]);
}

void minus_vector(real* sub_out,real* min, int len)
{
    for(int i =0; i<len; i++)
        sub_out[i]= min[i]-sub_out[i];
}

void set_vector(real *dst, real* ori, int len){
    for(int i =0; i<len; i++)
        dst[i]= ori[i];
}
void model_init(Configuration conf, sp_3matrix * model,
                sp_3matrix * weight, sp_matrix ** images, sp_imatrix * mask,
                sp_matrix *x_coordinates, sp_matrix *y_coordinates, sp_matrix *z_coordinates){
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(rng, rand());
    if (conf.initial_model == initial_model_uniform) {
        /* Set every pixel in the model to a random number between
         zero and one (will be normalized later) */
        create_initial_model_uniform(model, rng);
    } else if (conf.initial_model == initial_model_radial_average) {
        /* The model falls off raially in the same way as the average of
         all the patterns. On each pixel, some randomness is added with
         the strengh of conf.initial_modle_noise.*[initial pixel value] */
        create_initial_model_radial_average(model, images, conf.number_of_images, mask, conf.initial_model_noise, rng);
    } else if (conf.initial_model == initial_model_random_orientations) {
        /* Assemble the model from the given diffraction patterns
         with a random orientation assigned to each. */
        create_initial_model_random_orientations(model, weight, images, conf.number_of_images, mask, x_coordinates,
                                                 y_coordinates, z_coordinates, rng);
    }else if (conf.initial_model == initial_model_file) {
        /* Read the initial model from file.*/
        create_initial_model_file(model, conf.initial_model_file);
    } else if (conf.initial_model == initial_model_given_orientations) {
        /* Assemble the model from the given diffraction patterns
         with a rotation assigned from the file provided in
         conf.initial_rotations_file. */
        create_initial_model_given_orientations(model, weight, images, conf.number_of_images, mask, x_coordinates,
                                                y_coordinates, z_coordinates, conf.initial_rotations_file);
    }

}

/* Create the compressed model: model and the model
   weights used in the compress step: weight. */
 void create_initial_model_uniform(sp_3matrix *model, gsl_rng *rng) {
  const int N_model = sp_3matrix_size(model);
  for (int i = 0; i < N_model; i++) {
    model->data[i] = gsl_rng_uniform(rng);
  }
}

/* The model falls off raially in the same way as the average of
   all the patterns. On each pixel, some randomness is added with
   the strengh of conf.initial_modle_noise.*[initial pixel value] */
void create_initial_model_radial_average(sp_3matrix *model, sp_matrix **images,  int N_images,
                        sp_imatrix *mask, real initial_model_noise,
                        gsl_rng *rng) {
  const int model_side = sp_3matrix_x(model);
  /* Setup for calculating radial average */
  real *radavg = (real*) malloc(model_side/2*sizeof(real));
  int *radavg_count = (int*) malloc(model_side/2*sizeof(int));
  int r;
  for (int i = 0; i < model_side/2; i++) {
    radavg[i] = 0.0;
    radavg_count[i] = 0;
  }

  /* Calculate the radial average */
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int x = 0; x < model_side; x++) {
      for (int y = 0; y < model_side; y++) {
    if (sp_imatrix_get(mask, x, y) > 0 && sp_matrix_get(images[i_image], x, y) >= 0.) {
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
void reset_to_zero(void * vec, int len, size_t n){
    memset(vec, 0, len*n );
}

void devide_part(real* data, int len, int part){
    if(part <=0 )
        part =1;
    for(int i = 0; i<len; i++)
        data[i]/=part;
}

real calcualte_image_max(sp_imatrix * mask, sp_matrix** images, int N_images, int N_2d){
    /* Write preprocessed images 1: Calculate the maximum of all
     images. This is used for normalization of the outputed pngs.*/
    real image_max = 0.;
    #pragma omp parallel
    for (int i_image = 0; i_image < N_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i] == 1 && images[i_image]->data[i] > image_max) {
                image_max = images[i_image]->data[i];
            }
        }
    }
    return image_max;
}

int calculate_image_included(const Configuration conf, int N_images){
    if (conf.calculate_r_free) {
        return (1.-conf.r_free_ratio) * (float) N_images;
    } else {
        return N_images;
    }
}
unsigned long int get_seed(Configuration conf){
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
    return seed;
}

void calculate_distance_spmatrix(sp_matrix* radius, Configuration conf){
    for (int x = 0; x < conf.model_side; x++) {
        for (int y = 0; y < conf.model_side; y++) {
            sp_matrix_set(radius,x,y,sqrt(pow((real) x - conf.model_side/2.0 + 0.5, 2) +
                                          pow((real) y - conf.model_side/2.0 + 0.5, 2)));
        }
    }
}

void set_int_array(int * array, int value, int len){
#pragma omp parallel
    for(int i =0; i<len; i++)
        array[i] = value;
}
void set_real_array(real * array, real value, int len){
#pragma omp parallel
    for(int i =0; i<len; i++)
        array[i] = value;

}

void compute_best_rotations(real* full_respons, int N_images, int N_slices, int* best_rotation){
    real this_resp;
    real highest_resp;
    for (int i_image = 0; i_image < N_images; i_image++) {
        best_rotation[i_image] = 0;
        highest_resp = full_respons[0*N_images+i_image];
        for (int i_slice = 1; i_slice < N_slices; i_slice++) {
            this_resp = full_respons[i_slice*N_images+i_image];
            if (this_resp > highest_resp) {
                best_rotation[i_image] = i_slice;
                highest_resp = this_resp;
            }
        }
    }
}


void compute_best_respons(real* respons, int N_images, int N_slices, real* best_respons){
    for (int i_image = 0; i_image < N_images; i_image++) {
        best_respons[i_image] = respons[0*N_images+i_image];
#pragma omp parallel for
        for (int i_slice = 1; i_slice < N_slices; i_slice++) {
            if (!isnan(respons[i_slice*N_images+i_image]) && (respons[i_slice*N_images+i_image] > best_respons[i_image] || isnan(best_respons[i_image]))) {
                best_respons[i_image] = respons[i_slice*N_images+i_image];
            }
        }
        if (isnan(best_respons[i_image])) {
            printf("%d: best resp is nan\n", i_image);
#pragma omp parallel for
            for (int i_slice = 0; i_slice < N_slices; i_slice++) {
                if (!isnan(respons[i_slice*N_images+i_image])) {
                    printf("tot resp is bad but single is good\n");
                }
            }
        }
    }

}





