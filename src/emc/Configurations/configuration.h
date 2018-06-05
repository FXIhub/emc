#ifndef CONFIGURATION_H
#define CONFIGURATION_H
#pragma once
#include <spimage.h>
#include <libconfig.h>
#include <unistd.h>

//the diffraction model. true _poisson: observed pattern K~Poissrnd(\phi*W),
// where \phi is fluence and W is 2d slices
enum diff_type {true_poisson=0,
                absolute=1,
                poisson=2,
                relative=3,
                annealing_poisson=4,
                };

// 3d model initial method.
enum initial_model_type {initial_model_uniform=0,
                         initial_model_radial_average,
                         initial_model_random_orientations,
                         initial_model_file,
                         initial_model_given_orientations};

// Structure for reading configuration file.
typedef struct{

    int chunk_size;
    int random_seed;

    // for emc input & output
    int model_side; // 3d model size
    int image_binning; //average of image_binning*image_binning pixels
    int number_of_images;
    int compact_output;

    //for EMC computations:General
    int normalize_images;
    int recover_scaling;
    int exclude_images;
    int blur_model;
    int calculate_r_free;

    //for EMC computations: Gaussian model
    //int rotations_n;
    int number_of_iterations;
    int sigma_half_life;

    //Added 2016-10-24 by Jing Liu
    int isDebug;
    int calculate_fit;

    int isSochasticEMCon;
    double rotation_sample_ratio;
    // for detect configuration
    double wavelength;
    double pixel_size;
    double detector_distance;
    double initial_model_noise;
    double exclude_images_ratio;
    double blur_model_sigma;
    double r_free_ratio;
    double sigma_start;
    double sigma_final;



    const char *rotations_file;
    const char *mask_file;
    const char *image_prefix;
    const char *initial_model_file;
    const char *initial_rotations_file;
    const char *output_dir;

    diff_type diff;
    initial_model_type initial_model;


    const char* less_binning_mask_file;
    int less_binning;
    int less_binning_model_side;
    int detector_size;
    int isEarlyStopOn;
    int isLessBinningOn;
    double early_stop_threshold;
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
