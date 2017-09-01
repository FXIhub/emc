#include "configuration.h"
 
int read_configuration_file(const char *filename, Configuration *config_out)
{
  if (access(filename, F_OK) == -1) {
    return 0;
  }
  config_t config;
  config_init(&config);
  if (!config_read_file(&config,filename)) {
    fprintf(stderr,"%d - %s\n",
	   config_error_line(&config),
	   config_error_text(&config));
    config_destroy(&config);
    exit(1);
  }
  config_lookup_int(&config,"model_side",&(config_out->model_side));
  config_lookup_int(&config,"image_binning",&(config_out->image_binning));
  config_lookup_float(&config,"wavelength",&(config_out->wavelength));
  config_lookup_float(&config,"pixel_size",&(config_out->pixel_size));
  config_lookup_float(&config,"detector_distance",&(config_out->detector_distance));
  config_lookup_string(&config,"rotations_file",&(config_out->rotations_file));
  config_lookup_float(&config,"sigma_start",&(config_out->sigma_start));
  config_lookup_float(&config,"sigma_final",&(config_out->sigma_final));
  config_lookup_int(&config,"sigma_half_life",&(config_out->sigma_half_life));
  config_lookup_int(&config,"chunk_size",&(config_out->chunk_size));
  config_lookup_int(&config,"number_of_images",&(config_out->number_of_images));
  config_lookup_int(&config,"number_of_iterations",&(config_out->number_of_iterations));
  config_lookup_string(&config,"mask_file",&(config_out->mask_file));
  config_lookup_string(&config,"image_prefix",&(config_out->image_prefix));
  config_lookup_bool(&config,"normalize_images",&(config_out->normalize_images));
  config_lookup_bool(&config,"recover_scaling",&(config_out->recover_scaling));
  config_lookup_bool(&config,"calculate_fit",&(config_out->calculate_fit));

  const char *initial_model_string = (const char*)malloc(20*sizeof(char));
  config_lookup_string(&config,"initial_model",&initial_model_string);
  if (strcmp(initial_model_string, "uniform") == 0) {
    config_out->initial_model = initial_model_uniform;
  } else if (strcmp(initial_model_string, "radial average") == 0) {
    config_out->initial_model = initial_model_radial_average;
  } else if (strcmp(initial_model_string, "random orientations") == 0) {
    config_out->initial_model = initial_model_random_orientations;
  } else if (strcmp(initial_model_string, "file") == 0) {
    config_out->initial_model = initial_model_file;
  } else if (strcmp(initial_model_string, "given orientations") == 0) {
    config_out->initial_model = initial_model_given_orientations;
  } else {
    printf("Configuration file: bad value for initial_model: %s\n", initial_model_string);
    return 0;
  }
  config_lookup_float(&config,"initial_model_noise",&(config_out->initial_model_noise));
  config_lookup_string(&config,"initial_model_file",&(config_out->initial_model_file));
  config_lookup_string(&config, "initial_rotations_file", &(config_out->initial_rotations_file));
  config_lookup_bool(&config,"exclude_images",&(config_out->exclude_images));
  config_lookup_float(&config,"exclude_images_ratio",&(config_out->exclude_images_ratio));
  config_lookup_string(&config, "output_dir", &(config_out->output_dir));
  config_lookup_bool(&config, "calculate_r_free", &(config_out->calculate_r_free));
  config_lookup_float(&config, "r_free_ratio", &(config_out->r_free_ratio));
  const char *diff_type_string = (const char*)malloc(20*sizeof(char));
  config_lookup_string(&config,"diff_type",&diff_type_string);
  if (strcmp(diff_type_string, "absolute") == 0) {
    config_out->diff = absolute;
  } else if (strcmp(diff_type_string, "poisson") == 0) {
    config_out->diff = poisson;
  } else if (strcmp(diff_type_string, "relative") == 0) {
    config_out->diff = relative;
  } else if (strcmp(diff_type_string, "annealing_poisson") == 0) {
    config_out->diff = annealing_poisson;
  }else if (strcmp(diff_type_string, "true_poisson") == 0) {
      config_out->diff = true_poisson;
  }
  else {
    printf("Configuration file: bad value for diff_type: %s\n", diff_type_string);
    return 0;
  }
  config_lookup_int(&config,"blur_model",&(config_out->blur_model));
  config_lookup_float(&config,"blur_model_sigma",&(config_out->blur_model_sigma));
  // If set to zero then read the seed from /dev/random, otherwise
  // set it to the value specified
  config_out->random_seed = 0;
  config_lookup_int(&config, "random_seed", &(config_out->random_seed));
  config_lookup_int(&config, "compact_output", &(config_out->compact_output));
  config_out->pixel_size *= config_out->image_binning;
  config_lookup_int(&config, "isDebug", &(config_out->isDebug));
  return 1;
}

/* Dummy function at the moment */
int write_configuration_file(const char *filename, const Configuration config_in) {
  //FILE *output_stream = fopen(filename, 'wp');
  //config
  config_t config;
  config_init(&config);
  return 0;
}

/*
void get_distributed_config( Configuration conf, ConfigD* d_conf){
    (*d_conf).blur_model =conf.blur_model;
    (*d_conf).blur_model_sigma = conf.blur_model_sigma;
    (*d_conf).calculate_r_free = conf.calculate_r_free;
    (*d_conf).chunk_size = conf.chunk_size;
    (*d_conf).detector_distance = conf.detector_distance;
    (*d_conf).diff = (int) conf.diff;
    (*d_conf).exclude_images = conf.exclude_images;
    (*d_conf).exclude_images_ratio = conf.exclude_images_ratio;
    (*d_conf).model_side = conf.model_side;
    (*d_conf).number_of_images = conf.number_of_images;
    (*d_conf).number_of_iterations = conf.number_of_iterations;
    (*d_conf).pixel_size = conf.pixel_size;
    (*d_conf).recover_scaling = conf.recover_scaling;
    (*d_conf).r_free_ratio = conf.r_free_ratio;
    (*d_conf).sigma_final =conf.sigma_final;
    (*d_conf).sigma_half_life = conf.sigma_half_life;
    (*d_conf).sigma_start = conf.sigma_start;
    (*d_conf).wavelength = conf.wavelength;
    (*d_conf).isDebug = conf.isDebug;
}
 */
