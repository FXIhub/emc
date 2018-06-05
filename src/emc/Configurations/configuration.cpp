#include "configuration.h"

int getInt(const char * name, config_t * config)
{
    const config_setting_t *s = config_lookup(config, name);
    return config_setting_get_int(s);
}

double getFloat(const char * name, config_t * config)
{
    const config_setting_t *s = config_lookup(config, name);
    return config_setting_get_float(s);
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

    config_out->sigma_half_life = getInt("sigma_half_life", &config);
    config_out->number_of_iterations = getInt("number_of_iterations", &config);
    config_out->random_seed = getInt("random_seed",&config);
    config_out->model_side = getInt("model_side",&config);
    config_out->image_binning = getInt("image_binning",&config);
    config_out->chunk_size = getInt("chunk_size",&config);
    config_out->number_of_images = getInt("number_of_images",&config);
    config_out->compact_output = getInt("compact_output",&config);
    config_out->blur_model = getInt("blur_model",&config);
    config_out->isDebug = getInt("isDebug",&config);
    config_out->calculate_fit = getInt("calculate_fit",&config);
    config_out->calculate_r_free = getInt("calculate_r_free",&config);
    config_out->normalize_images = getInt("normalize_images",&config);
    config_out->recover_scaling = getInt("recover_scaling",&config);
    config_out->exclude_images = getInt("exclude_images",&config);

    config_out->isSochasticEMCon = getInt("isSochasticEMCon",&config);
    config_out->rotation_sample_ratio = getFloat("rotation_sample_ratio",&config);

    config_out->random_seed = 0;
    config_out->wavelength = getFloat("wavelength",&config);
    config_out->pixel_size = getFloat("pixel_size",&config) * config_out->image_binning;
    config_out->detector_distance = getFloat("detector_distance",&config);
    config_out->sigma_start = getFloat("sigma_start",&config);
    config_out->sigma_final = getFloat("sigma_final",&config);
    config_out->initial_model_noise = getFloat("initial_model_noise",&config);
    config_out->exclude_images_ratio = getFloat("exclude_images_ratio",&config);
    config_out->r_free_ratio = getFloat("r_free_ratio",&config);
    config_out->blur_model_sigma = getFloat("blur_model_sigma",&config);



    config_lookup_string(&config,"rotations_file",&(config_out->rotations_file));
    config_lookup_string(&config,"mask_file",&(config_out->mask_file));
    config_lookup_string(&config,"image_prefix",&(config_out->image_prefix));
    config_lookup_string(&config,"initial_model_file",&(config_out->initial_model_file));
    config_lookup_string(&config, "initial_rotations_file", &(config_out->initial_rotations_file));
    config_lookup_string(&config, "output_dir", &(config_out->output_dir));


    const char* tmp_string;

    config_lookup_string(&config,"diff_type",&tmp_string);
    if (strcmp(tmp_string, "absolute") == 0) {
        config_out->diff = absolute;
    } else if (strcmp(tmp_string, "poisson") == 0) {
        config_out->diff = poisson;
    } else if (strcmp(tmp_string, "relative") == 0) {
        config_out->diff = relative;
    } else if (strcmp(tmp_string, "annealing_poisson") == 0) {
        config_out->diff = annealing_poisson;
    }else if (strcmp(tmp_string, "true_poisson") == 0) {
        config_out->diff = true_poisson;
    }
    else {
        printf("Configuration file: bad value for diff_type: %s\n", tmp_string);
        return 0;
    }

    config_lookup_string(&config,"initial_model",(const char**)&tmp_string);
    if (strcmp(tmp_string, "uniform") == 0) {
        config_out->initial_model = initial_model_uniform;
    } else if (strcmp(tmp_string, "radial average") == 0) {
        config_out->initial_model = initial_model_radial_average;
    } else if (strcmp(tmp_string, "random orientations") == 0) {
        config_out->initial_model = initial_model_random_orientations;
    } else if (strcmp(tmp_string, "file") == 0) {
        config_out->initial_model = initial_model_file;
    } else if (strcmp(tmp_string, "given orientations") == 0) {
        config_out->initial_model = initial_model_given_orientations;
    } else {
        printf("Configuration file: bad value for initial_model: %s\n", tmp_string);
        return 0;
    }

    // printf("conf diff type %d\n\n", config_out->diff);

    config_lookup_string(&config,"less_binning_mask_file",&(config_out->less_binning_mask_file));
    config_out->less_binning = getInt("less_binning",&config);
    config_out->detector_size = getInt("detector_size",&config);
    config_out->less_binning_model_side = getInt("less_binning_model_side",&config);
    config_out->isEarlyStopOn = getInt("isEarlyStopOn",&config);
    config_out->isLessBinningOn = getInt("isLessBinningOn",&config);
    config_out->early_stop_threshold = getFloat("early_stop_threshold",&config);



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
