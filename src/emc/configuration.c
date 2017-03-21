#include "configuration.h"

void init_configuration(Configuration *config) {
  config->rotations_file = malloc(MAX_STRING_LENGTH*sizeof(char));
  config->mask_file = malloc(MAX_STRING_LENGTH*sizeof(char));
  config->image_prefix = malloc(MAX_STRING_LENGTH*sizeof(char));
  config->initial_model_file = malloc(MAX_STRING_LENGTH*sizeof(char));
  config->initial_rotations_file = malloc(MAX_STRING_LENGTH*sizeof(char));
  config->output_dir = malloc(MAX_STRING_LENGTH*sizeof(char));
}

void missing_variable(char *message) {
  printf("Configuration file must contain value for %s", message);
  exit(1);
}

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
  if (!config_lookup_int(&config,"model_side",&(config_out->model_side)))
    missing_variable("model_side");
  config_lookup_int(&config,"image_binning",&(config_out->image_binning));
  if (!config_lookup_float(&config,"wavelength",&(config_out->wavelength)))
    missing_variable("wavelength");
  if (!config_lookup_float(&config,"pixel_size",&(config_out->pixel_size)))
    missing_variable("pixel_size");
  if (!config_lookup_float(&config,"detector_distance",&(config_out->detector_distance)))
    missing_variable("detector_distance");
  if (!config_lookup_string(&config,"rotations_file",&config_out->rotations_file))
    missing_variable("rotations_file");
  /* if (config_lookup(&config, "rotations_file")) { */
  /*   strcpy(config_out->rotations_file, config_lookup_string(&config, "rotations_file")); */
  /* } else { */
  /*   missing_variable("rotations_file"); */
  /* } */
  if (!config_lookup_float(&config,"sigma_start",&(config_out->sigma_start)))
    missing_variable("sigma_start");
  config_lookup_float(&config,"sigma_final",&(config_out->sigma_final));
  config_lookup_int(&config,"sigma_half_life",&(config_out->sigma_half_life));
  config_lookup_int(&config,"chunk_size",&(config_out->chunk_size));
  if (!config_lookup_int(&config,"number_of_images",&(config_out->number_of_images)))
    missing_variable("number_of_images");
  config_lookup_int(&config,"number_of_iterations",&(config_out->number_of_iterations));
  config_lookup_bool(&config,"individual_masks",&(config_out->individual_masks));
  config_lookup_string(&config,"mask_file",&(config_out->mask_file));
  if (!config_lookup_string(&config,"image_prefix",&(config_out->image_prefix)))
    missing_variable("image_prefix");
  config_lookup_bool(&config,"normalize_images",&(config_out->normalize_images));
  config_lookup_bool(&config,"recover_scaling",&(config_out->recover_scaling));
  const char *initial_model_string = malloc(20*sizeof(char));
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
  const char *diff_type_string = malloc(20*sizeof(char));
  config_lookup_string(&config,"diff_type",&diff_type_string);
  if (strcmp(diff_type_string, "absolute") == 0) {
    config_out->diff = absolute;
  } else if (strcmp(diff_type_string, "poisson") == 0) {
    config_out->diff = poisson;
  } else if (strcmp(diff_type_string, "relative") == 0) {
    config_out->diff = relative;
  } else if (strcmp(diff_type_string, "annealing_poisson") == 0) {
    config_out->diff = annealing_poisson;
  } else {
    printf("Configuration file: bad value for diff_type: %s\n", diff_type_string);
    return 0;
  }
  config_lookup_int(&config,"blur_model",&(config_out->blur_model));
  config_lookup_float(&config,"blur_model_sigma",&(config_out->blur_model_sigma));

  // If set to zero then read the seed from /dev/random, otherwise
  // set it to the value specified
  config_out->random_seed = 0;
  config_lookup_int(&config, "random_seed", &(config_out->random_seed));
  config_lookup_bool(&config, "compact_output", &(config_out->compact_output));

  config_out->pixel_size *= config_out->image_binning;
  return 1;
}

void create_default_config(Configuration *config) {
  config->model_side = 128; // no
  config->image_binning = 1;
  config->wavelength = 1e-9; //no
  config->pixel_size = 75e-9; //no
  config->detector_distance = 0.74; //no
  //const char *rotations_file; //no
  config->sigma_start = 0.1; //no
  config->sigma_final = config->sigma_start;
  config->sigma_half_life = 50;
  config->diff = poisson;
  config->chunk_size = 1000;
  config->number_of_images = 1000; //no
  config->number_of_iterations = 100;
  config->individual_masks = 0;
  strcpy(config->mask_file, "");
  //const char *image_prefix; //no
  config->normalize_images = 1;
  config->recover_scaling = 1;
  config->initial_model = initial_model_radial_average;
  config->initial_model_noise = 0.01;
  strcpy(config->initial_model_file, "");
  strcpy(config->initial_rotations_file, "");
  config->exclude_images = 0;
  config->exclude_images_ratio = 0.0;
  config->blur_model = 0;
  config->blur_model_sigma = 3.;
  strcpy(config->output_dir, "./output");
  config->random_seed = 0;
  config->calculate_r_free = 0;
  config->r_free_ratio = 0.1;
  config->compact_output = 1;
}

/* Dummy function at the moment */
int write_configuration_file(const char *filename, const Configuration *config_in) {
  //FILE *output_stream = fopen(filename, 'wp');
  //config
  config_t config;
  config_init(&config);

  config_setting_t *root = config_root_setting(&config);
  config_setting_t *setting;

  setting = config_setting_add(root, "model_side", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->model_side);

  setting = config_setting_add(root, "image_binning", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->image_binning);

  setting = config_setting_add(root, "wavelength", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->wavelength);

  setting = config_setting_add(root, "pixel_size", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->pixel_size);

  setting = config_setting_add(root, "detector_distance", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->detector_distance);

  setting = config_setting_add(root, "rotations_file", CONFIG_TYPE_STRING);
  config_setting_set_string(setting, config_in->rotations_file);

  setting = config_setting_add(root, "sigma_start", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->sigma_start);

  setting = config_setting_add(root, "sigma_final", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->sigma_final);

  setting = config_setting_add(root, "sigma_half_life", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->sigma_half_life);

  setting = config_setting_add(root, "diff_type", CONFIG_TYPE_STRING);
  if (config_in->diff == absolute) {
    config_setting_set_string(setting, "absolute");
  } else if (config_in->diff == poisson) {
    config_setting_set_string(setting, "poisson");
  } else if (config_in->diff == relative) {
    config_setting_set_string(setting, "relative");
  } else if (config_in->diff == annealing_poisson) {
    config_setting_set_string(setting, "annealing_poisson");
  }

  setting = config_setting_add(root, "chunk_size", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->chunk_size);

  setting = config_setting_add(root, "number_of_images", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->number_of_images);

  setting = config_setting_add(root, "number_of_iterations", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->number_of_iterations);

  setting = config_setting_add(root, "individual_masks", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->individual_masks);

  setting = config_setting_add(root, "mask_file", CONFIG_TYPE_STRING);
  config_setting_set_string(setting, config_in->mask_file);

  setting = config_setting_add(root, "image_prefix", CONFIG_TYPE_STRING);
  config_setting_set_string(setting, config_in->image_prefix);

  setting = config_setting_add(root, "normalize_images", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->normalize_images);

  setting = config_setting_add(root, "recover_scaling", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->recover_scaling);

  setting = config_setting_add(root, "initial_model", CONFIG_TYPE_STRING);
  if (config_in->initial_model == initial_model_uniform) {
    config_setting_set_string(setting, "uniform");
  } else if (config_in->initial_model == initial_model_radial_average) {
    config_setting_set_string(setting, "radial average");
  } else if (config_in->initial_model == initial_model_random_orientations) {
    config_setting_set_string(setting, "random orientations");
  } else if (config_in->initial_model == initial_model_file) {
    config_setting_set_string(setting, "file");
  } else if (config_in->initial_model == initial_model_given_orientations) {
    config_setting_set_string(setting, "given orientations");
  }

  setting = config_setting_add(root, "initial_model_noise", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->initial_model_noise);

  setting = config_setting_add(root, "initial_model_file", CONFIG_TYPE_STRING);
  config_setting_set_string(setting, config_in->initial_model_file);

  setting = config_setting_add(root, "initial_rotations_file", CONFIG_TYPE_STRING);
  config_setting_set_string(setting, config_in->initial_rotations_file);

  setting = config_setting_add(root, "exclude_images", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->exclude_images);

  setting = config_setting_add(root, "exclude_images_ratio", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->exclude_images_ratio);

  setting = config_setting_add(root, "blur_model", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->blur_model);

  setting = config_setting_add(root, "blur_model_sigma", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->blur_model_sigma);

  setting = config_setting_add(root, "output_dir", CONFIG_TYPE_STRING);
  config_setting_set_string(setting, config_in->output_dir);

  setting = config_setting_add(root, "random_seed", CONFIG_TYPE_INT);
  config_setting_set_int(setting, config_in->random_seed);

  setting = config_setting_add(root, "calculate_r_free", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->calculate_r_free);

  setting = config_setting_add(root, "r_free_ratio", CONFIG_TYPE_FLOAT);
  config_setting_set_float(setting, config_in->r_free_ratio);

  setting = config_setting_add(root, "compact_output", CONFIG_TYPE_BOOL);
  config_setting_set_bool(setting, config_in->compact_output);

  config_write_file(&config, filename);
  return 0;
}
