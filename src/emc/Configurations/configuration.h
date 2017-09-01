#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#pragma once
#include <spimage.h>
#include <libconfig.h>
#include <unistd.h>

//the diffraction model. true _poisson: observed pattern K~Poissrnd(\phi*W),
// where \phi is fluence and W is 2d slices
enum diff_type {absolute=0,
                poisson,
                relative,
                annealing_poisson,
                true_poisson};

// 3d model initial method.
enum initial_model_type {initial_model_uniform=0,
                         initial_model_radial_average,
                         initial_model_random_orientations,
                         initial_model_file,
                         initial_model_given_orientations};

// Structure for reading configuration file.
typedef struct{
    // for detect configuration
    double wavelength;
    double pixel_size;
    double detector_distance;

    // for emc input & output
    int model_side; // 3d model size
    int image_binning; //average of image_binning*image_binning pixels
    const char *rotations_file;
    int number_of_images;
    const char *mask_file;
    const char *image_prefix;
    enum initial_model_type initial_model;
    double initial_model_noise;
    const char *initial_model_file;
    const char *initial_rotations_file;
    const char *output_dir;
    int compact_output;

    //for EMC computations:General
    enum diff_type diff;
    int chunk_size;
    int number_of_iterations;
    int normalize_images;
    int recover_scaling;
    int exclude_images;
    double exclude_images_ratio;
    int blur_model;
    double blur_model_sigma;
    int random_seed;
    int calculate_r_free;
    double r_free_ratio;

    //for EMC computations: Gaussian model
    //int rotations_n;
    double sigma_start;
    double sigma_final;
    int sigma_half_life;

    //Added 2016-10-24 by Jing Liu
    int isDebug;
    int calculate_fit;

}Configuration;

int read_configuration_file(const char *filename, Configuration *config_out);

int write_configuration_file(const char *filename, const Configuration config);


// Distributed configuration structure, if MPI is used
/*typedef struct{
    double wavelength; // wave length of xray, got from cheetah
    double pixel_size; // magnification of image, got from cheetah
    double detector_distance; // the distance between detector, got from
    double sigma_start;
    double sigma_final;
    double exclude_images_ratio;
    double blur_model_sigma;
    double r_free_ratio;

    int model_side;    // the size of model is model_size * model_size
    int chunk_size;
    int number_of_images; // the number of diffraction images.
    int number_of_iterations; // the maximal number of iterations

    int sigma_half_life;
    int diff;
    int recover_scaling;
    int exclude_images; // defines if any images should be excluded from
    int blur_model;
    int calculate_r_free;
    int isDebug;
    
}ConfigD;

void get_distributed_config(Configuration, ConfigD*);
*/


#endif
