/* this file generates frames given a model and rotations
*/

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <emc.h>
#include <mpi.h>
#include <MPIhelper.h>
#include <FILEhelper.h>
#include <emc_math.h>
#include <TIMERhelper.h>
using namespace std;

#include <string.h>


int main(int argc, char **argv)
{
    Configuration conf;
    /*-----------------------------------------------------------Do Master Node Initial Job start------------------------------------------------*/
    cout<<"Init MPI...";
    MPI_Init(&argc, &argv);
    int taskid, ntasks;
    int master = 0;
    unsigned long int timeB = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
    printf("%d of total %d MPI processes started!\n", taskid,ntasks);
    if(taskid == master) cuda_print_device_info();
    if (argc > 1) {
        read_configuration_file(argv[1],&conf);
    } else {
        read_configuration_file("./emc.conf",&conf);
    }
    const int N_images = conf.number_of_images;
    int slice_chunk = conf.chunk_size;
    int N_2d = conf.model_side*conf.model_side;
    int N_model = conf.model_side*conf.model_side*conf.model_side;

    real*  model_data = (real*)malloc(N_model*sizeof(real));
    real*  model_weight_data = (real*)malloc(N_model*sizeof(real));
    real* modelP  =(real*)malloc(N_model*sizeof(real));
    sp_imatrix **masks = (sp_imatrix **)  malloc(N_images*sizeof(sp_imatrix *));
    sp_matrix **images = read_images(conf,masks);
    sp_imatrix * mask = read_mask(conf);
    sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    real central_part_radius = 10;
    sp_3matrix *model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
    sp_3matrix *model_weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);

    //normalize_images(images, mask, conf, central_part_radius);
    real image_max = calcualte_image_max( mask,  images, N_images, N_2d);
    calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength, x_coordinates, y_coordinates, z_coordinates);
    model_init(conf,model, model_weight,images,mask, x_coordinates, y_coordinates, z_coordinates);
    Quaternion *rotations;
    real *weights_rotation;
    const int N_slices = read_rotations_file(conf.rotations_file, &rotations, &weights_rotation);

    /*------------------------------task devision respons MPI BUFFER------------------------------------*/
    int *lens = (int *)malloc(sizeof(int)*ntasks);
    int slice_start = taskid* N_slices/ntasks;
    int slice_backup = slice_start;
    int slice_end =  (taskid+1)* N_slices/ntasks;
    int* recvcounts = (int *)malloc(sizeof(int)*ntasks);
    int* dispal = (int*) malloc(sizeof(int) *ntasks);
    dispal[0] = 0;
    if (taskid == ntasks -1) slice_end = N_slices;
    int allocate_slice = slice_end - slice_backup;
    for (int i = 0; i<ntasks-1; i++){
        lens[i] =  get_allocate_len(ntasks,N_slices,i);
        recvcounts[i] = lens[i]*N_images;
        dispal[i+1] = lens[i]*N_images + dispal[i];
    }
    lens[ntasks-1] =  get_allocate_len(ntasks,N_slices,ntasks-1);
    recvcounts[ntasks-1] = lens[ntasks-1]*N_images;

    real* d_sum;
    cuda_allocate_real(&d_sum,N_images);

    real h_sum_vector[N_images];
    real* sum_vector = &h_sum_vector[0];

    real h_maxr[N_images];
    real* maxr = &h_maxr[0];
    real tmpbuf_images[N_images];//MPI recv buff
    real* tmpbuf_images_ptr = &tmpbuf_images[0];
    int index[N_images];//MPI recv buff
    int* index_ptr = &index[0];

    real* d_maxr;
    cuda_allocate_real(&d_maxr,N_images);

    /*------------------------------task devision of respons  MPI BUFFER------------------------------------*/

    /* Get a random seed from /dev/random or from the
     configuration file if provided. */
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
    unsigned long int seed = get_seed(conf);
    srand(seed);
    gsl_rng_set(rng, rand());

    int N_images_included=calculate_image_included(conf, N_images);


    /* Allocate spimage object used for outputting the model.*/
    Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
    for (int i = 0; i < N_model; i++) {
        model_out->image->data[i] = sp_cinit(model->data[i],0.0);
        if (model_weight->data[i] > 0.0) {
            model_out->mask->data[i] = 1;
        } else {
            model_out->mask->data[i] = 0;
        }
    }

    char filename_buffer[PATH_MAX];
    sprintf(filename_buffer,"%s/model_init.h5", conf.output_dir);
    sp_image_write(model_out,filename_buffer,0);
    for (int i = 0; i < N_model; i++) {
        model_out->image->data[i] = sp_cinit(model_weight->data[i],0.0);
    }
    sprintf(filename_buffer,"%s/model_init_weight.h5", conf.output_dir);
    sp_image_write(model_out,filename_buffer,0);

    /* Create the matrix radius where the value of each pixel
     is the distance to the center. */
    sp_matrix *radius = sp_matrix_alloc(conf.model_side,conf.model_side);
    calculate_distance_spmatrix(radius,conf);
    /* Create and initialize the scaling variables on the CPU. */
    real *scaling = (real*) malloc(N_images*allocate_slice*sizeof(real));
    set_real_array(scaling,1.0, N_images*allocate_slice);
    int *active_images = (int*) malloc(N_images*sizeof(int));
    set_int_array(active_images,1, N_images);
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
    real *respons = (real*) malloc(allocate_slice*N_images*sizeof(real));
    real total_respons = 0;

    /*----------------  GPU VAR INIT START     -------------------------------*/

    real * d_slices;
    cuda_allocate_slices(&d_slices,conf.model_side,slice_chunk);
    real * d_model;
    cuda_allocate_model(&d_model,model);
    real * d_model_updated;
    cuda_allocate_model(&d_model_updated,model);
    real * d_model_tmp;
    real * d_model_weight; //used to d_weight
    cuda_allocate_model(&d_model_weight,model_weight);


    /* List of all sampled rotations. Used in both expansion and
     compression. Does not change. */
    real *d_weights_rotation;
    cuda_allocate_real(&d_weights_rotation, allocate_slice);
    cuda_copy_weight_to_device(weights_rotation, d_weights_rotation, allocate_slice,taskid);
    real * d_rotations;
    cuda_allocate_rotations_chunk(&d_rotations,rotations,slice_start,slice_end);

    real * d_x_coord;
    real * d_y_coord;
    real * d_z_coord;
    cuda_allocate_coords(&d_x_coord, &d_y_coord, &d_z_coord, x_coordinates,
                         y_coordinates,  z_coordinates);
    int * d_mask;
    cuda_allocate_mask(&d_mask,mask);

    real * d_images;
    cuda_allocate_images(&d_images,images,N_images);
    cuda_apply_single_mask(d_images, d_mask, N_2d, N_images);

    /* Individual masks read from each diffraction pattern.
     Only used for the last iteration. */
    int * d_masks;
    cuda_allocate_masks(&d_masks, masks, N_images);

    /* Array of all diffraction patterns with mask applied. */
    real * d_images_individual_mask;
    cuda_allocate_images(&d_images_individual_mask, images, N_images);
    cuda_apply_masks(d_images, d_masks, N_2d, N_images);

    /* Responsability matrix local*/
    real * d_respons;
    cuda_allocate_real(&d_respons,allocate_slice*N_images);

    //scaling
    real * d_scaling;
    cuda_allocate_real(&d_scaling,N_images*allocate_slice);
    cuda_copy_real_to_device(scaling,d_scaling, N_images*allocate_slice);
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
    real *radial_fit = (real*) malloc(round(conf.model_side/2)*sizeof(real));
    real *radial_fit_weight = (real*) malloc(round(conf.model_side/2)*sizeof(real));
    real *d_radial_fit;
    real *d_radial_fit_weight;
    cuda_allocate_real(&d_radial_fit, round(conf.model_side/2));
    cuda_allocate_real(&d_radial_fit_weight,round(conf.model_side/2));
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
    sprintf(filename_buffer, "%s/state.h5", conf.output_dir);
    hid_t state_file = open_state_file(filename_buffer);
    FILE *timeFile = create_file_descriptor(conf, "/exeTime.data", "wp");
    FILE *likelihood = create_file_descriptor(conf, "/likelihood.data", "wp");
    FILE *best_rot_file = create_file_descriptor(conf,"/best_rot.data", "wp");
    FILE *best_quat_file;
    FILE *fit_file = create_file_descriptor(conf,"/fit.data", "wp");
    FILE *fit_best_rot_file = create_file_descriptor(conf,"/fit_best_rot.data", "wp");
    FILE *radial_fit_file = create_file_descriptor(conf,"/radial_fit.data", "wp");
    FILE *r_free;
    if (conf.calculate_r_free) {
        r_free =create_file_descriptor(conf,"/r_free.data","wp");
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
    real *d_weight_map;
    cuda_allocate_weight_map(&d_weight_map, conf.model_side);
    real weight_map_radius, weight_map_falloff;
    real weight_map_radius_start = conf.model_side; // Set start radius to contain entire pattern
    real weight_map_radius_final = conf.model_side; // Set final radius to contain entire pattern


    real sigma;
    int current_chunk;
    int start_iteration =0;

    // for distributed edition
    real * full_scaling= (real*) malloc(N_images* N_slices *sizeof(real));
    real * full_respons= (real*) malloc(N_images* N_slices *sizeof(real));

    cuda_reset_real(d_respons, N_images*allocate_slice);
    real *average_resp = (real*) malloc(N_slices*sizeof(real));
    real mean = cuda_model_average(d_model,N_model);
    printf("model average is %f \n",mean);
    /*int rots[198] = {6,20,48,48,685,723,834,957,1182,1411,1958,1975,2213,2267,2358,2548,2848,2855,2856,3479,4013,4288,4408,4531,4661,4680,4852,4900,5049,5050,6293,6392,9397,10640,10796,13072,15630,15795,16394,18667,18686,19225,19660,19845,20047,20514,22272,22774,23621,23655,24683,24854,25499,29437,30704,31703,31868,32042,33181,33234,34299,35819,36553,37165,38023,38811,46152,46187,46845,49528,52449,52832,53220,53222,54788,55441,56663,57877,59783,60253,60789,61566,63506,63676,65534,68413,69664,70348,70944,71157,73586,74235,74441,75277,75390,75451,75637,78499,79383,80264,80266,80480,80757,80790,82294,82487,82708,84319,84349,85807,85826,86161,88255,89871,89905,91992,92346,93532,93563,94603,95133,95817,96314,97165,97595,101764,105009,105304,105611,106486,108955,108958,108961,108987,111401,113035,119224,121163,124050,127700,132909,134660,138579,145169,147137,148168,155010,155844,158511,169444,169904,172467,177049,191873,198444,201268,201571,209972,215570,216092,216124,221736,221817,227076,229658,229658,229699,230635,248987,249058,274251,281142,285710,311733,312598,312990,325605,343545,353279,353771,355230,357875,359511,365869,367684,371352,373018,376333,378596,380820,381642,387249,389530,389741,389755,389861,390777,396671};*/
    /*int rots [198] = {6,20,48,48,685,723,834,957,1182,1411,1958,1975,2213,2267,2358,2548,2848,2855,2856,3479,4013,4288,4408,4531,4661,4680,4852,4900,5049,5050,6293,6392,9397,10640,10796,13072,15630,15795,16394,18667,18686,19225,19660,19845,20047,20514,22272,22774,23621,23655,24683,24854,25499,29437,30704,31703,31868,32042,33181,33234,34299,35819,36553,37165,38023,38811,46152,46187,46845,49528,52449,52832,53220,53222,54788,55441,56663,57877,59783,60253,60789,61566,63506,63676,65534,68413,69664,70348,70944,71157,73586,74235,74441,75277,75390,75451,75637,78499,79383,80264,80266,80480,80757,80790,82294,82487,82708,84319,84349,85807,85826,86161,88255,89871,89905,91992,92346,93532,93563,94603,95133,95817,96314,97165,97595,101764,105009,105304,105611,106486,108955,108958,108961,108987,111401,113035,119224,121163,124050,127700,132909,134660,138579,145169,147137,148168,155010,155844,158511,169444,169904,172467,177049,191873,198444,201268,201571,209972,215570,216092,216124,221736,221817,227076,229658,229658,229699,230635,248987,249058,274251,281142,285710,311733,312598,312990,325605,343545,353279,353771,355230,357875,359511,365869,367684,371352,373018,376333,378596,380820,381642,387249,389530,389741,389755,389861,390777,396671};*/
    //	for (int i =0; i< 198; i++){

    //	}
    real* slices = (real*) malloc(sizeof(real) * N_slices*N_2d);
    cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord, 0, N_images);
    cuda_copy_real_to_host(slices,d_slices,N_images*N_2d);
    Image *write_image = sp_image_alloc(conf.model_side, conf.model_side, 1);
    for (int i_image = 0; i_image < N_images; i_image++) {
        for (int i = 0; i < N_2d; i++) {
            if (mask->data[i]) {
                sp_real(write_image->image->data[i]) = slices[i+i_image*N_2d];
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
}


