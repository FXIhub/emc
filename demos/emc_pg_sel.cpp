#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <emc.h>
#include <signal.h>
#include <sys/stat.h>
#include <hdf5.h>
#include <getopt.h>
#include <time.h>

//#define PATH_MAX 256

static int quit_requested = 0;


int main(int argc, char **argv)
{
    /* Parse command-line options */
    char configuration_filename[PATH_MAX] = "emc_pg.conf";
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
    /* Read the configuration file */
    Configuration conf;
    int conf_return = read_configuration_file(configuration_filename, &conf);
    if (conf_return == 0)
        error_exit_with_message("Can't read configuration file %s\nRun emc -h for help.", configuration_filename);

    /* This buffer is used for names of all output files */
    char filename_buffer[PATH_MAX];

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
    sp_imatrix **masks = (sp_imatrix **)  malloc(conf.number_of_images*sizeof(sp_imatrix *));
    sp_matrix **images = read_images(conf,masks);
    /*
  sp_matrix **images = read_images_cxi("/home/ekeberg/Data/LCLS_SPI/2015July/cxi/narrow_filter_normalized.cxi",
                       "/entry_1/data", "/entry_1/mask", conf.number_of_images, conf.model_side,
                       conf.image_binning, masks);
  */
    sp_imatrix * mask = read_mask(conf);
/*
    if (conf.normalize_images) {
        if (!conf.recover_scaling) {
            normalize_images_preserve_scaling(images, mask, conf);
        } else {
            //real central_part_radius = 0.; // Zero here means that the entire image is used.
            real central_part_radius = 10.; // Zero here means that the entire image is used.
            normalize_images_central_part(images, mask, central_part_radius, conf);
        }
    }
*/
    /* Write preprocessed images 1: Calculate the maximum of all
     images. This is used for normalization of the outputed pngs.*/
    real image_max = 0.;
    for (int i_image = 0; i_image < N_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i] == 1 && images[i_image]->data[i] > image_max) {
                image_max = images[i_image]->data[i];
            }
        }
    }

    /* Write preprocessed images 2: Create the spimage images to
     output. Including putting in the general mask.*/
    Image *write_image = sp_image_alloc(conf.model_side, conf.model_side, 1);
    for (int i_image = 0; i_image < N_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i]) {
                sp_real(write_image->image->data[i]) = images[i_image]->data[i];
            } else {
                sp_real(write_image->image->data[i]) = 0.0;
            }
            write_image->mask->data[i] = mask->data[i];
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
        create_initial_model_radial_average(model, images, N_images, mask, conf.initial_model_noise, rng);

    } else if (conf.initial_model == initial_model_random_orientations) {
        /* Assemble the model from the given diffraction patterns
       with a random orientation assigned to each. */
        create_initial_model_random_orientations(model, weight, images, N_images, mask, x_coordinates,
                                                 y_coordinates, z_coordinates, rng);
    }else if (conf.initial_model == initial_model_file) {
        /* Read the initial model from file.*/
        create_initial_model_file(model, conf.initial_model_file);
    } else if (conf.initial_model == initial_model_given_orientations) {
        /* Assemble the model from the given diffraction patterns
       with a rotation assigned from the file provided in
       conf.initial_rotations_file. */
        create_initial_model_given_orientations(model, weight, images, N_images, mask, x_coordinates,
                                                y_coordinates, z_coordinates, conf.initial_rotations_file);
    }

    /* Allocate spimage object used for outputting the model.*/
    Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
    for (int i = 0; i < N_model; i++) {
        model_out->image->data[i] = sp_cinit(model->data[i],0.0);
        if (weight->data[i] > 0.0) {
            model_out->mask->data[i] = 1;
        } else {
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
    real *scaling = (real*) malloc(N_images*N_slices*sizeof(real));
    for (int i = 0; i < N_images*N_slices; i++) {
        scaling[i] = 1.0;
    }

    /* Create active images and initialize them. If calculate_r_free
     is used the active_images variable keeps track of which
     images are included an which are excluded. */
    int *active_images = (int*) malloc(N_images*sizeof(int));
    for (int i_image = 0; i_image < N_images; i_image++) {
        active_images[i_image] = 1;
    }
    if (conf.calculate_r_free) {
        int *index_list = (int*) malloc(N_images*sizeof(int));
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
    real *respons = (real*) malloc(N_slices*N_images*sizeof(real));
    real total_respons;
    real *average_resp = (real*) malloc(N_slices*sizeof(real));

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
    int * d_mask;
    cuda_allocate_mask(&d_mask, mask);

    /* Array of all diffraction patterns. */
    real * d_images;
    cuda_allocate_images(&d_images, images, N_images);
    cuda_apply_single_mask(d_images, d_mask, N_2d, N_images);

    /* Individual masks read from each diffraction pattern.
     Only used for the last iteration. */
    int * d_masks;
    cuda_allocate_masks(&d_masks, masks, N_images);

    /* Array of all diffraction patterns with mask applied. */
    real * d_images_individual_mask;
    cuda_allocate_images(&d_images_individual_mask, images, N_images);
    cuda_apply_masks(d_images, d_masks, N_2d, N_images);

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
    real *fit = (real*) malloc(N_images*sizeof(real));
    real *d_fit;
    cuda_allocate_real(&d_fit,N_images);

    /* fit_best_rot is like fit but instead ov weighted average
     over all orientations, each image is just considered in its
     best fitting orientation. */
    real *fit_best_rot = (real*) malloc(N_images*sizeof(real));
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
    real *radial_fit = (real*) malloc(conf.model_side/2*sizeof(real));
    real *radial_fit_weight = (real*) malloc(conf.model_side/2*sizeof(real));
    real *d_radial_fit;
    real *d_radial_fit_weight;
    cuda_allocate_real(&d_radial_fit, conf.model_side/2);
    cuda_allocate_real(&d_radial_fit_weight, conf.model_side/2);
    real* best_respons = (real*) malloc(N_images*sizeof(real));

    /* best_rotation stores the index of the rotation with the
     highest responsability for each diffraction pattern. */
    int *best_rotation = (int*) malloc(N_images*sizeof(int));
    int *d_best_rotation;
    cuda_allocate_int(&d_best_rotation, N_images);
    //from jing
    real *d_best_respons;
    cuda_allocate_real(&d_best_respons, N_images);

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

    /* Start the main EMC loop */
    for (int iteration = 0; iteration < conf.number_of_iterations; iteration++) {
        /* If ctrl-c was pressed execution stops but the final
       iteration using individual masks and cleenup still runs. */
        if (quit_requested == 1) {
            break;
        }
        printf("\niteration %d\n", iteration);

        real sum = cuda_model_average(d_model, conf.model_side*conf.model_side*conf.model_side);
        printf("model average is %f \n ", sum);

        /* Sigma is a variable that describes the noise that is
       typically either constant or decreasing on every iteration. */
        sigma = conf.sigma_final + (conf.sigma_start-conf.sigma_final)*exp(-iteration/(float)conf.sigma_half_life*log(2.));
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
        cuda_set_to_zero(d_fit,N_images);
        cuda_set_to_zero(d_radial_fit,conf.model_side/2);
        cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2);

        /* Find and output the best orientation for each diffraction pattern,
       i.e. the one with the highest responsability. */
            //cuda_set_to_zero(d_respons,N_images*N_slices%);
            //reset_real(best_respons,N_images);

        cuda_calculate_best_rotation(d_respons,d_best_respons, d_best_rotation, N_images, N_slices);
        cuda_copy_int_to_host(best_rotation, d_best_rotation, N_images);
        cuda_copy_real_to_host(best_respons,d_best_respons, N_images);
        cuda_copy_int_to_host(best_rotation,d_best_rotation, N_images);
        
        

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
        
        real* slices = (real*) malloc(sizeof(real) * N_slices*N_2d);
        printf("allocate slices\n");

        /* In this loop through the chunks the "fit" is calculated */
        for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= N_slices) {
                current_chunk = N_slices - slice_start;
            } else {
                current_chunk = slice_chunk;
            }
            cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord, slice_start, current_chunk);
            printf("copy real to host slices, %d\n", slice_start);
            cuda_copy_real_to_host(&slices[slice_start*N_2d],d_slices, slice_chunk * N_2d);


            /* Calculate the "fit" between the diffraction patterns
     and the compressed model. There are two versions of this:
     one weighted average and one where only the best orientation
     of each pattern is considered. */
  
          cuda_calculate_fit(d_slices, d_images, d_mask, d_scaling,
                               d_respons, d_fit, sigma, N_2d, N_images,
                               slice_start, current_chunk);
            cuda_calculate_fit_best_rot(d_slices, d_images, d_mask, d_scaling,
                                        d_best_rotation, d_fit_best_rot, N_2d, N_images,
                                        slice_start, current_chunk);
            /* Calculate a radially averaged version of the weightd "fit" */
            if (iteration % radial_fit_n == 0 && iteration != 0) {
                cuda_calculate_radial_fit(d_slices, d_images, d_mask,
                                          d_scaling, d_respons, d_radial_fit,
                                          d_radial_fit_weight, d_radius,
                                          N_2d, conf.model_side, N_images, slice_start,
                                          current_chunk);
 
          }
        }
        
int rots[198] = {6,20,48,48,685,723,834,957,1182,1411,1958,1975,2213,2267,2358,2548,2848,2855,2856,3479,4013,4288,4408,4531,4661,4680,4852,4900,5049,5050,6293,6392,9397,10640,10796,13072,15630,15795,16394,18667,18686,19225,19660,19845,20047,20514,22272,22774,23621,23655,24683,24854,25499,29437,30704,31703,31868,32042,33181,33234,34299,35819,36553,37165,38023,38811,46152,46187,46845,49528,52449,52832,53220,53222,54788,55441,56663,57877,59783,60253,60789,61566,63506,63676,65534,68413,69664,70348,70944,71157,73586,74235,74441,75277,75390,75451,75637,78499,79383,80264,80266,80480,80757,80790,82294,82487,82708,84319,84349,85807,85826,86161,88255,89871,89905,91992,92346,93532,93563,94603,95133,95817,96314,97165,97595,101764,105009,105304,105611,106486,108955,108958,108961,108987,111401,113035,119224,121163,124050,127700,132909,134660,138579,145169,147137,148168,155010,155844,158511,169444,169904,172467,177049,191873,198444,201268,201571,209972,215570,216092,216124,221736,221817,227076,229658,229658,229699,230635,248987,249058,274251,281142,285710,311733,312598,312990,325605,343545,353279,353771,355230,357875,359511,365869,367684,371352,373018,376333,378596,380820,381642,387249,389530,389741,389755,389861,390777,396671};
	/*int rots [198] = {6,20,48,48,685,723,834,957,1182,1411,1958,1975,2213,2267,2358,2548,2848,2855,2856,3479,4013,4288,4408,4531,4661,4680,4852,4900,5049,5050,6293,6392,9397,10640,10796,13072,15630,15795,16394,18667,18686,19225,19660,19845,20047,20514,22272,22774,23621,23655,24683,24854,25499,29437,30704,31703,31868,32042,33181,33234,34299,35819,36553,37165,38023,38811,46152,46187,46845,49528,52449,52832,53220,53222,54788,55441,56663,57877,59783,60253,60789,61566,63506,63676,65534,68413,69664,70348,70944,71157,73586,74235,74441,75277,75390,75451,75637,78499,79383,80264,80266,80480,80757,80790,82294,82487,82708,84319,84349,85807,85826,86161,88255,89871,89905,91992,92346,93532,93563,94603,95133,95817,96314,97165,97595,101764,105009,105304,105611,106486,108955,108958,108961,108987,111401,113035,119224,121163,124050,127700,132909,134660,138579,145169,147137,148168,155010,155844,158511,169444,169904,172467,177049,191873,198444,201268,201571,209972,215570,216092,216124,221736,221817,227076,229658,229658,229699,230635,248987,249058,274251,281142,285710,311733,312598,312990,325605,343545,353279,353771,355230,357875,359511,365869,367684,371352,373018,376333,378596,380820,381642,387249,389530,389741,389755,389861,390777,396671};*/
//	for (int i =0; i< 198; i++){
	
//	}

 Image *write_image = sp_image_alloc(conf.model_side, conf.model_side, 1);
    for (int i_image = 0; i_image < N_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i]) {
                sp_real(write_image->image->data[i]) = slices[i+ rots[i_image]*N_2d];
            } else {
                sp_real(write_image->image->data[i]) = 0.0;
            }
            write_image->mask->data[i] = mask->data[i];
        }
	sprintf(filename_buffer, "%s/image_%.4d.h5", conf.output_dir, i_image);
        sp_image_write(write_image, filename_buffer, 0);

        /* Set a corner pixel to image_max. This assures correct relative scaling of pngs. */
        write_image->image->data[0] = sp_cinit(image_max, 0.);
        sprintf(filename_buffer, "%s/image_%.4d.png", conf.output_dir, i_image);
        sp_image_write(write_image, filename_buffer, SpColormapJet|SpColormapLogScale);
    }
    sp_image_free(write_image);
exit(0);

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
                cuda_update_scaling_full(d_images, d_slices, d_mask, d_scaling, d_weight_map, N_2d, N_images, slice_start, current_chunk, conf.diff);
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

            cuda_get_slices(model,d_model,d_slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,
                            slice_start,current_chunk);

            cuda_calculate_responsabilities(d_slices, d_images, d_mask, d_weight_map,
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
            real *best_respons = (real*) malloc(N_images*sizeof(real));
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

        /* Reset the compressed model */
        cuda_reset_model(model,d_model_updated);
        cuda_reset_model(weight,d_weight);

        /* Exclude images. Use the assumption that the diffraction patterns
       with the lowest maximum responsability does not belong in the data.
       Therefore these are excluded from the compression step. */
        real *best_respons;
        if (conf.exclude_images == 1 && iteration > -1) {
            /* Find the highest responsability for each diffraction pattern */
            best_respons = (real*) malloc(N_images*sizeof(real));
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
            real *best_respons_copy = (real*) malloc(N_images_included*sizeof(real));
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
       //debug reset active_images to 1s
        if (iteration == 0) {
            for (int i_image = 0; i_image < N_images; i_image++) {
                active_images[i_image] = 1;
            }
        }
        cuda_copy_int_to_device(active_images, d_active_images, N_images);

        /* Start update scaling second time (test) */
        if (conf.recover_scaling) {
            for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
                if (slice_start + slice_chunk >= N_slices) {
                    current_chunk = N_slices - slice_start;
                } else {
                    current_chunk = slice_chunk;
                }
                cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
                                slice_start, current_chunk);

                cuda_update_scaling_full(d_images, d_slices, d_mask, d_scaling, d_weight_map, N_2d, N_images, slice_start, current_chunk, conf.diff);
            }
        }
        /* End update scaling second time (test) */

        /* This loop through the slice chunks updates compresses the model. */
        for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= N_slices) {
                current_chunk = N_slices - slice_start;
            } else {
                current_chunk = slice_chunk;
            }

            /* This function does both recalculate part of the expanded model
     and compresses this part. The model needs to be divided with
     the weights outside this loop. */
            cuda_update_slices(d_images, d_slices, d_mask,
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

        cuda_copy_model(model, d_model);
        cuda_copy_model(weight, d_weight);
        if(conf.isDebug){
            printf("debug ... model %f %f %f %f \n,\n" ,model->data[32*17*19],model->data[32*16*18],
                    weight->data[32*17*19], weight->data[32*16*18]);
        }
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
        if (conf.blur_model) {
            cuda_blur_model(d_model, conf.model_side, conf.blur_model_sigma);
        }

        /* Copy the new compressed model to the CPU. */
        cuda_copy_model(model, d_model);
        cuda_copy_model(weight, d_weight);
        if(conf.isDebug){
            printf("debug ... model %f %f %f %f \n,\n" ,model->data[32*17*19],model->data[32*16*18],
                    weight->data[32*17*19], weight->data[32*16*18]);
        }

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
        cuda_update_slices_final(d_images_individual_mask, d_slices, d_mask,
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
