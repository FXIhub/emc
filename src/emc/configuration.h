#pragma once
#include <spimage.h>
#include <libconfig.h>

#ifdef __cplusplus
extern "C"{
#endif

  enum diff_type {absolute=0, poisson, relative};

   typedef struct{
    int model_side;
    int read_stride;
    double wavelength;
    double pixel_size;
    double detector_distance;
    //int rotations_n;
    const char *rotations_file;
    double sigma_start;
    double sigma_final;
    int sigma_half_life;
    enum diff_type diff;
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
    double initial_model_noise;
    const char *model_file;
    const char *init_rotations_file;
    int exclude_images;
    double exclude_ratio;
    double model_blur;
    const char *output_dir;
    const char *debug_dir;
    int random_seed;
    int calculate_r_free;
    double r_free_ratio;
    int compact_output;
  }Configuration;

  Configuration read_configuration_file(const char *filename);

  int write_configuration_file(const char *filename, const Configuration config);

#ifdef __cplusplus
  }
#endif
