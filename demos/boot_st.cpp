/*
 * Author : Jing Liu@ Biophysics and TDB
 * Modified to collective reduce, to C version
 * 2016-10-25 merged to Tomas single GPU verstion
 */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
//#ifdef IN_PLACE
//#undef IN_PLACE
//#undef BOTTOM
//#endif
#include <emc.h>
#include <mpi.h>
#include <MPIhelper.h>
#include <FILEhelper.h>
#include <emc_math.h>
#include <TIMERhelper.h>
using namespace std;

#include <string.h>

#define  MPI_EMC_PRECISION MPI_FLOAT
#define N_bootstrapping 100
int main(int argc, char *argv[]){
    Configuration conf;
    Configuration conf_copy;
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
        read_configuration_file(argv[1],&conf_copy);
    } else {
        read_configuration_file("./emc.conf",&conf);
        read_configuration_file(argv[1],&conf_copy);
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
    sp_matrix **all_images = read_images(conf,masks);
    sp_imatrix * mask = read_mask(conf);
    sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
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
    /*if (conf.recover_scaling) {
        cuda_normalize_model(model, d_model);
        cuda_normalize_model(model, d_model_updated);
    }*/

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
    //cuda_apply_single_mask(d_images, d_mask, N_2d, N_images); // used to be apply_single_mask


    /* Individual masks read from each diffraction pattern.
     Only used for the last iteration. */
    int * d_masks;
    cuda_allocate_masks(&d_masks, masks, N_images);

    /* Array of all diffraction patterns with mask applied. */
    real * d_images_individual_mask;
    cuda_allocate_images(&d_images_individual_mask, images, N_images);
    //cuda_apply_masks(d_images, d_masks, N_2d, N_images);

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
    //cuda_allocate_real(&d_fit,N_images);

    /* fit_best_rot is like fit but instead ov weighted average
     over all orientations, each image is just considered in its
     best fitting orientation. */
    real *fit_best_rot = (real*) malloc(N_images*sizeof(real));
    real *d_fit_best_rot;
    //cuda_allocate_real(&d_fit_best_rot, N_images);

    /* If calculate_r_free is used the active_images variable keeps
     track of which images are included an which are excluded. */
    int *d_active_images;
    cuda_allocate_int(&d_active_images,N_images);
    cuda_copy_int_to_device(active_images, d_active_images, N_images);

    /* 3D array where each value is the distance to the center
     of that pixel. */
    real *d_radius;
    //    cuda_allocate_real(&d_radius, N_2d);
    //   cuda_copy_real_to_device(radius->data, d_radius, N_2d);


    /* Radial fit is the same as fit but instead of as a function
     of diffraction pattern index it is presented as a function
     of distance to the center. */
    real *radial_fit = (real*) malloc(round(conf.model_side/2)*sizeof(real));
    real *radial_fit_weight = (real*) malloc(round(conf.model_side/2)*sizeof(real));
    real *d_radial_fit;
    real *d_radial_fit_weight;
    //cuda_allocate_real(&d_radial_fit, round(conf.model_side/2));
    //cuda_allocate_real(&d_radial_fit_weight,round(conf.model_side/2));
    real* best_respons = (real*) malloc(N_images*sizeof(real));

    /* best_rotation stores the index of the rotation with the
     highest responsability for each diffraction pattern. */
    int *best_rotation = (int*) malloc(N_images*sizeof(int));
    int *d_best_rotation;
    //cuda_allocate_int(&d_best_rotation, N_images);
    //from jing
    real *d_best_respons;
    //cuda_allocate_real(&d_best_respons, N_images);

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
    int* permute = (int*) malloc(sizeof(int)*conf.number_of_images);
    /*-----------------------------------------------------------Do local init END-----------------------------------------------------------*/

    /*------------------------------------ EMC Iterations start ----------------------------------------------------------*/
    for (int run = 0; run<N_bootstrapping; run++){
        if(taskid ==master && run >0){
            for(int i =0; i <N_images; i++){
                int a =floor(drand48() *N_images);
                // make sure that a is within [0,N_images-1]
                if (a < 0)
                    a=0;
                if (a >= N_images)
                    a = N_images -1;
                cout << a<<" ";
                memcpy(images[i]->data, all_images[a]->data, sizeof(real)*N_2d);
                images[i]->rows=all_images[a]->rows;
                images[i]->cols=all_images[a]->cols;
                permute[i] = a;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        Broadcast_Images(images,conf.number_of_images,master);

        cuda_copy_images_2_device(&d_images,images,N_images);
        model_init(conf,model, model_weight,images,mask, x_coordinates, y_coordinates, z_coordinates);
        cuda_copy_model_2_device(&d_model,model);
        MPI_Barrier(MPI_COMM_WORLD);


        for (int iteration = start_iteration; iteration < conf.number_of_iterations; iteration++) {
            /*---------------------- reset local variable-------------------------------*/
            if(taskid == master){
                if (iteration == start_iteration )
                    timeB = gettimenow();
                else{
                    double exeTime =  update_time(timeB, gettimenow());
                    write_time(timeFile,  exeTime,  iteration-1);
                }
            }
            cuda_reset_real(d_sum, N_images);
            cuda_reset_real(d_maxr, N_images);
            reset_to_zero(tmpbuf_images,N_images, sizeof(real));
            reset_to_zero(maxr, N_images, sizeof(real));
            reset_to_zero(sum_vector,N_images, sizeof(real));

            /*---------------------- reset local variable done-------------------------*/

            /* Sigma is a variable that describes the noise that is
         typically either constant or decreasing on every iteration. */
            sigma = conf.sigma_final + (conf.sigma_start-conf.sigma_final)*exp(-iteration/(float)conf.sigma_half_life*log(2.));
            real sum = cuda_model_average(d_model,N_model);

            if(taskid == master){
                printf("\niteration %d\n", iteration);
                printf("model average is %f \n ", sum);
                printf("sigma = %g\n", sigma);
            }
            //from tomas
            /* Calculate the weightmap radius for this particular iteration. */
            weight_map_radius = weight_map_radius_start + ((weight_map_radius_final-weight_map_radius_start) *
                                                           ((real)iteration / ((real)conf.sigma_half_life)));
            weight_map_falloff = 0.;
            int radial_fit_n = 1; // Allow less frequent output of the fit by changing this output period
            //cuda_set_to_zero(d_fit,N_images);
            //cuda_set_to_zero(d_radial_fit,conf.model_side/2);
            //cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2);

            /* Find and output the best orientation for each diffraction pattern,
         i.e. the one with the highest responsability. */
            if (iteration == start_iteration && strcmp(conf.initial_rotations_file, "not used") ==0){
                reset_to_zero(respons,N_images*allocate_slice, sizeof(real));
                cuda_set_to_zero(d_respons, N_images*allocate_slice);
                cuda_set_to_zero(d_slices,N_2d*slice_chunk);
                cuda_copy_real_to_device(scaling,d_scaling, N_images*allocate_slice);
                real* nom = (real* ) malloc(sizeof(real) *N_images);
                real* den = (real* ) malloc(sizeof(real) *N_images);
                real* d_nom;
                real* d_den;
                cuda_allocate_real(&d_nom,N_images);
                cuda_allocate_real(&d_den,N_images);
                cuda_set_to_zero(d_nom,N_images);
                cuda_set_to_zero(d_den,N_images);
            }


            /*---------------------- Calculate scaling start -------------------------*/
            //printf(" cal scaling! \n\n");

            if (conf.recover_scaling) {
                //printf("recover scaling \n\n");
                for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
                    if (slice_start + slice_chunk >= slice_end) {
                        current_chunk = slice_end - slice_start;
                    } else {
                        current_chunk = slice_chunk;
                    }
                    int current_start = slice_start- slice_backup;
                    cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
                                    current_start   , current_chunk);
                    cuda_update_scaling_full(d_images, d_slices, d_mask, d_scaling, d_weight_map,
                                             N_2d, N_images, current_start, current_chunk, diff_type(conf.diff));

                }

                //printf(" write scaling! \n\n");
                cuda_copy_real_to_host(scaling, d_scaling, N_images*allocate_slice);
                cuda_copy_real_to_host(respons, d_respons, allocate_slice*N_images);
                MPI_Barrier(MPI_COMM_WORLD);
                if(conf.compact_output == 0 ){
                    Global_Gatherv((void*)scaling, recvcounts[taskid],
                                   MPI_EMC_PRECISION, (void*) full_scaling,
                                   recvcounts, dispal, master, MPI_COMM_WORLD);
                    if(taskid ==master){
                        if(conf.isDebug == 1)
                            printf("DEBUG... %g %g %g %g \n\n", full_scaling[0],full_scaling[100],
                                    full_scaling[N_slices*N_images-2],full_scaling[N_slices*N_images-100]);
                        sprintf(filename_buffer, "%s/scaling_%.4d.h5", conf.output_dir, iteration);
                        write_2d_real_array_hdf5(filename_buffer, full_scaling, N_slices, N_images);
                    }
                }
                /* Output the best scaling */
                Global_Gatherv((void*)respons, recvcounts[taskid],
                               MPI_EMC_PRECISION, (void*) full_respons,
                               recvcounts, dispal, master, MPI_COMM_WORLD);
                if(taskid == master)
                    write_scaling_to_file(scaling_dataset, iteration, full_scaling, full_respons, N_slices);

            }
            else cuda_copy_real_to_device(scaling,d_scaling,allocate_slice*N_images);

            /*---------------------- Calculate scaling end -------------------------*/

            /*-----------------------------start respons normalization (distribution ) start-------------------------------*/

            cuda_set_to_zero(d_respons,N_images*allocate_slice);
            /* In this loop through the chunks the responsabilities are updated. */
            for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
                if (slice_start + slice_chunk >= slice_end) {
                    current_chunk = slice_end - slice_start;
                } else {
                    current_chunk = slice_chunk;
                }
                int current_start = slice_start- slice_backup;
                cuda_get_slices(model,d_model,d_slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,
                                current_start,current_chunk);
                cuda_calculate_responsabilities(d_slices, d_images, d_mask, d_weight_map,
                                                sigma, d_scaling, d_respons, d_weights_rotation,
                                                N_2d, N_images, current_start,
                                                current_chunk,  diff_type(conf.diff));

            }
            MPI_Barrier(MPI_COMM_WORLD);
            Global_Gatherv((void*)respons, recvcounts[taskid],
                           MPI_EMC_PRECISION, (void*) full_respons,
                           recvcounts,  dispal, master, MPI_COMM_WORLD);

            //cuda_copy_real_to_host(respons,d_respons,N_slices*N_images);
            cuda_max_vector(d_respons, N_images, allocate_slice,d_maxr);
            cuda_copy_real_to_host(maxr,d_maxr,N_images);
            MPI_Barrier(MPI_COMM_WORLD);
            Global_Allreduce(maxr, tmpbuf_images,N_images,MPI_EMC_PRECISION,MPI_MAX, MPI_COMM_WORLD);
            cuda_copy_real_to_device(tmpbuf_images, d_maxr, N_images);
            cuda_respons_max_expf(d_respons,d_maxr,N_images, allocate_slice, d_sum);
            total_respons = cuda_total_respons( d_respons,allocate_slice*N_images);

            cuda_copy_real_to_host(sum_vector,d_sum,N_images);
            MPI_Barrier(MPI_COMM_WORLD);
            Global_Allreduce(sum_vector,tmpbuf_images_ptr,N_images,MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
            cuda_copy_real_to_device(tmpbuf_images_ptr,d_sum,N_images);
            cuda_norm_respons_sumexpf(d_respons,d_sum,maxr,N_images,allocate_slice);
            //total_respons = cuda_total_respons(d_respons,respons,N_images*allocate_slice);
            //cuda_copy_real_to_host(respons,d_respons,allocate_slice*N_images);
            MPI_Barrier(MPI_COMM_WORLD);
            Global_Gatherv((void*)respons, recvcounts[taskid],
                           MPI_EMC_PRECISION, (void*) full_respons,
                           recvcounts,  dispal, master, MPI_COMM_WORLD);
            Global_Allreduce(&total_respons, &total_respons,1, MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
            total_respons /= N_images*N_slices;
            if(taskid ==master){
                sprintf(filename_buffer, "%s/responsabilities_%.4d.h5", conf.output_dir, iteration);
                write_2d_real_array_hdf5_transpose(filename_buffer, full_respons, N_slices, N_images);
                write_ave_respons(conf,full_respons,N_images,N_slices, iteration, average_resp);
                write_real_array(likelihood, &total_respons,1);

            }
            if(conf.isDebug == 1){
                printf("DEBUG... TOT_probablity after normalization %g %g at taskid %d\n", total_respons, total_respons,taskid);
                printf("DEBUG... full_respons is  %f %f %f \n", full_respons[0], full_respons[N_images*allocate_slice-1], full_respons[N_images*N_slices-1]);
            }

            /* Reset the compressed model */
            cuda_reset_model(model,d_model_updated);
            cuda_reset_model(model_weight,d_model_weight);


            /*-----------------------------end respons normalization (distribution ) done-------------------------------*/

            /* start update model */
            cuda_reset_model(model,d_model_updated);
            cuda_reset_model(model_weight,d_model_weight);
            //cout <<"UPDATE MODEL!" <<endl;
            for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
                if (slice_start + slice_chunk >= slice_end) {
                    current_chunk = slice_end - slice_start;
                } else {
                    current_chunk = slice_chunk;
                }
                int current_start = slice_start- slice_backup;
                if(conf.diff != true_poisson)
                    cuda_update_slices(d_images, d_slices, d_mask,
                                       d_respons, d_scaling, d_active_images,
                                       N_images, current_start, current_chunk, N_2d,
                                       model,d_model_updated, d_x_coord, d_y_coord,
                                       d_z_coord, &d_rotations[current_start*4],
                            d_model_weight,images);
                else
                    cuda_update_true_poisson_slices(d_images, d_slices, d_mask,
                                                    d_respons, d_scaling, d_active_images,
                                                    N_images, current_start, current_chunk, N_2d,
                                                    model,d_model_updated, d_x_coord, d_y_coord,
                                                    d_z_coord, &d_rotations[current_start*4],
                            d_model_weight,images);
            }
            d_model_tmp = d_model_updated;
            d_model_updated = d_model;
            d_model = d_model_tmp;

            // collect model and model_weight back to the master node
            cuda_copy_model(model,d_model);
            cuda_copy_model(model_weight,d_model_weight);
            if (conf.isDebug == 1){
                sum = cuda_model_average(d_model,N_model);
                printf("model average is %g before adding up at %d\n", sum, taskid);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            Global_Allreduce(model->data,model_data, N_model, MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
            Global_Allreduce(model_weight->data,model_weight_data,N_model,MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            memcpy(model->data,model_data,sizeof(real)*N_model);
            memcpy(model_weight->data,model_weight_data,N_model*sizeof(real));
            cuda_copy_model_2_device(&d_model,model);
            cuda_copy_model_2_device(&d_model_weight,model_weight);

            if (conf.isDebug == 1){
                sum = cuda_model_average(d_model,N_model);
                printf("model average is %g after adding up at %d\n", sum, taskid);
            }
            /* When all slice chunks have been compressed we need to divide the
         model by the model weights. */
            cuda_divide_model_by_weight(model, d_model, d_model_weight);


            if (conf.recover_scaling){
                cuda_normalize_model_given_mean(model, d_model,mean);
            }
            if (conf.blur_model) {
                cuda_blur_model(d_model, conf.model_side, conf.blur_model_sigma);
            }
            cuda_copy_model(model,d_model);
            cuda_copy_model(model_weight,d_model_weight);

            MPI_Barrier(MPI_COMM_WORLD);

            if (conf.isEarlyStopOn &iteration-start_iteration>0)
            {
                cuda_copy_real_to_device(modelP, d_model_updated, N_model);
                real diff = cuda_model_diff(d_model,d_model_updated,N_model)/N_model;
                printf(" model different at iteration %d and %d is %f \n", iteration, iteration-1,diff);
                if(diff<conf.early_stop_thr){
                    if(taskid ==master){
                        write_model(conf, run, N_model, model,model_weight);
                        write_weight(conf, run, N_model,model_weight);
                        write_state_file_iteration(state_file, run);
                    }
                    break;
                }
            }
            memcpy(modelP,model->data,sizeof(real)*N_model);
        }//END FOR ITERATION
        /* Close files that have been open throughout the anlysis. */

        close_state_file(state_file);
        fclose(timeFile);
        fclose(likelihood);
        fclose(best_rot_file);
        fclose(fit_file);
        fclose(fit_best_rot_file);
        fclose(radial_fit_file);
        if (conf.calculate_r_free) {
            fclose(r_free);
        }
        if (conf.recover_scaling){
            close_scaling_file(scaling_dataset, scaling_file);
        }
    }

    MPI_Finalize();
    return 0;
}






