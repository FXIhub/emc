#pragma once
#include <spimage.h>
#include <libconfig.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"{
#endif

  enum diff_type {absolute=0,
		  poisson,
		  relative,
		  annealing_poisson};
  enum initial_model_type {initial_model_uniform=0,
			   initial_model_radial_average,
			   initial_model_random_orientations,
			   initial_model_file,
			   initial_model_given_orientations};

   typedef struct{
    int model_side;
    int image_binning;
    double wavelength;
    double pixel_size;
    double detector_distance;
    //int rotations_n;
    const char *rotations_file;
    double sigma_start;
    double sigma_final;
    int sigma_half_life;
    enum diff_type diff;
    int chunk_size;
    int number_of_images;
    int number_of_iterations;
    int individual_masks;
    const char *mask_file;
    const char *image_prefix;
    int normalize_images;
    int recover_scaling;
    enum initial_model_type initial_model;
    double initial_model_noise;
    const char *initial_model_file;
    const char *initial_rotations_file;
    int exclude_images;
    double exclude_images_ratio;
    int blur_model;
    double blur_model_sigma;
    const char *output_dir;
    int random_seed;
    int calculate_r_free;
    double r_free_ratio;
    int compact_output;
  }Configuration;

  int read_configuration_file(const char *filename, Configuration *config_out);

  int write_configuration_file(const char *filename, const Configuration config);

#ifdef __cplusplus
  }
#endif
