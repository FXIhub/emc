/*
 * Author : Jing Liu@ Biophysics and TDB
 * Modified to collective reduce, to C version
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
#include <unistd.h>

#include <string.h>

#define  MPI_EMC_PRECISION MPI_FLOAT
const int N_init_frames = 500;
const real error_lim =0.52;
const int N_step = 5;
const int N_max= 11;
const int N_reload = 2;
const char* logFile_path= "./logFile.text";
int main(int argc, char *argv[]){
    Configuration conf;
    /*-----------------------------------------------------------Do Master Node Initial Job start------------------------------------------------*/
    cout<<"Init MPI...";
    MPI_Init(&argc, &argv);
    int taskid, ntasks;
    int master = 0;
    unsigned long int timeB = 0;
    unsigned long int timestep = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
    printf("%d of total %d MPI processes started!\n", taskid,ntasks);
    if(taskid == master) cuda_print_device_info();
    if (argc > 1) {
        read_configuration_file(argv[1],&conf);
    } else {
        printf("reading config...");
        read_configuration_file("./emc.conf",&conf);
        printf("done");
    }
    real  model_data[conf.model_side*conf.model_side*conf.model_side];
    real  model_weight_data[conf.model_side*conf.model_side*conf.model_side];
    
    
    //sp_imatrix **masks = (sp_imatrix **)  malloc(conf.number_of_images*sizeof(sp_imatrix *));

    Quaternion *rotations;
    real *weights_rotation;
    const int N_slices = read_rotations_file(conf.rotations_file, &rotations, &weights_rotation);
    imageInd list; // init list 0-N_init_Frames;
    //sp_matrix **images = read_images(conf,masks);
    sp_imatrix * mask = read_mask(conf);
    sp_matrix **images = (sp_matrix**)malloc(conf.number_of_images*sizeof(sp_matrix *));
    sp_imatrix **masks = (sp_imatrix**)malloc(conf.number_of_images*sizeof(sp_imatrix *));
    for (int i = 0; i < conf.number_of_images; i++) {
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);
    }
    //init rea first N_init frames from logged files
    int N_acc =0;
    
    long int current_time =50;
    int reloaded =0;
        //for debug
    /*FILE* logFW = fopen(logFile_path,"w");
    for (int i =0; i<10; i++)
    fprintf(logFW, "%.4d %.8d %.4d %.1d %.4f\n", 3+i,3+i+1,103,1,0.5);
    fclose(logFW);*/
    FILE* logF = fopen(logFile_path,"r");
    
    //test
    if (logF == NULL) printf ("Error opening log file\n\n");
    else{
        while (N_acc <N_init_frames ){
            cout<<"reading"<<endl;
            // file_ind classtime rotation category error
        reloaded = load_selected_images_by_log (conf, logF, images,masks, list,  "%ld %ld %d %d %f\n","%s%.4d",
                                              error_lim, current_time,
                                              N_max-N_acc);
        printf("%d", reloaded);
            if (reloaded ==0) sleep(5);
            
        N_acc += reloaded;
        }
    }

    sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    real central_part_radius = 10;
    sp_3matrix *model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
    sp_3matrix *model_weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);

    const int N_images = conf.number_of_images;
    const int slice_chunk = conf.chunk_size;
    const int N_2d = conf.model_side*conf.model_side;
    const int N_model = conf.model_side*conf.model_side*conf.model_side;

    normalize_images(images, mask, conf, central_part_radius);
    calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength, x_coordinates, y_coordinates, z_coordinates);
    model_init(conf,model, model_weight,images,mask, x_coordinates, y_coordinates, z_coordinates);


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
    if (conf.recover_scaling) {
        cuda_normalize_model(model, d_model);
        cuda_normalize_model(model, d_model_updated);
    }

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
    int start_iteration =1;

    // for distributed edition
    real * full_scaling= (real*) malloc(N_images* N_slices *sizeof(real));
    real * full_respons= (real*) malloc(N_images* N_slices *sizeof(real));

    cuda_reset_real(d_respons, N_images*allocate_slice);
    real *average_resp = (real*) malloc(N_slices*sizeof(real));
    /*-----------------------------------------------------------Do local init END-----------------------------------------------------------*/

    /*------------------------------------ EMC Iterations start ----------------------------------------------------------*/
    //cout << "isDebug "<<conf.isDebug<< endl;
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

        /* Find and output the best orientation for each diffraction pattern,
         i.e. the one with the highest responsability. */
        if (iteration == start_iteration && strcmp(conf.initial_rotations_file, "not used") ==0){
            if(conf.isDebug == 1)
                printf("DEBUG... ROTFILE ISUSED%s %d\n\n", conf.initial_rotations_file,strcmp(conf.initial_rotations_file, "not used"));
            reset_to_zero(respons,N_images*allocate_slice, sizeof(real));
            cuda_set_to_zero(d_respons, N_images*allocate_slice);

            reset_to_zero(best_respons,N_images, sizeof(real));
            cuda_set_to_zero(d_best_respons,N_images);

            reset_to_zero(best_rotation,N_images, sizeof(int));
            cuda_copy_int_to_device(best_rotation, d_best_rotation, N_images);

            cuda_set_to_zero(d_slices,N_2d*slice_chunk);
             cuda_copy_real_to_device(scaling,d_scaling, N_images*allocate_slice);
        }
        else{
            reset_to_zero(best_respons,N_images, sizeof(real));
            cuda_calculate_best_rotation(d_respons, d_best_respons, d_best_rotation, N_images, allocate_slice);
            cuda_copy_real_to_host(best_respons,d_best_respons, N_images);
            cuda_copy_int_to_host(best_rotation,d_best_rotation, N_images);

            MPI_Barrier(MPI_COMM_WORLD);
            Global_Allreduce( best_respons, maxr,  N_images,
                              MPI_EMC_PRECISION, MPI_MAX, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i =0; i<N_images; i++){
                if (best_respons[i] != maxr[i])
                    best_rotation[i] = -10;
                else
                    best_rotation[i] += slice_backup;
            }
            memcpy(best_respons, maxr, sizeof(real) * N_images);
            MPI_Barrier(MPI_COMM_WORLD);
            Global_Allreduce( best_rotation,  index_ptr,  N_images,
                              MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            memcpy(best_rotation,index_ptr, sizeof(int) * N_images);
            cuda_copy_int_to_device(best_rotation, d_best_rotation, N_images);
             if(taskid ==master){
                write_int_array(best_rot_file, index_ptr, N_images);
                write_best_quat(conf, iteration,rotations, best_rotation, N_images);
            }
        }
        //real* nom = (real* ) malloc(sizeof(real) *N_images);
        //real* den = (real* ) malloc(sizeof(real) *N_images);
        real* d_nom;
        real* d_den;
        cuda_allocate_real(&d_nom,N_images);
        cuda_allocate_real(&d_den,N_images);
        cuda_set_to_zero(d_nom,N_images);
        cuda_set_to_zero(d_den,N_images);


        /*---------------------- Calculate scaling start -------------------------*/

        if (conf.recover_scaling) {
            printf("reconver scaling \n\n");
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
        total_respons = cuda_total_respons(d_respons,respons,N_images*allocate_slice);
        cuda_copy_real_to_host(respons,d_respons,allocate_slice*N_images);
        MPI_Barrier(MPI_COMM_WORLD);
        Global_Allreduce(&total_respons, &total_respons,1 ,MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
        Global_Gatherv((void*)respons, recvcounts[taskid],
                       MPI_EMC_PRECISION, (void*) full_respons,
                       recvcounts,  dispal, master, MPI_COMM_WORLD);

        /*-----------------------------start respons normalization (distribution ) start-------------------------------*/
        //cuda_copy_real_to_host(respons,d_respons,N_slices*N_images);
        cuda_max_vector(d_respons, N_images, allocate_slice,d_maxr);
        cuda_copy_real_to_host(maxr,d_maxr,N_images);
        MPI_Barrier(MPI_COMM_WORLD);
        Global_Allreduce(maxr, tmpbuf_images,N_images,MPI_EMC_PRECISION,MPI_MAX, MPI_COMM_WORLD);
        cuda_copy_real_to_device(tmpbuf_images, d_maxr, N_images);
        cuda_respons_max_expf(d_respons,d_maxr,N_images, allocate_slice, d_sum);
        cuda_copy_real_to_host(sum_vector,d_sum,N_images);
        MPI_Barrier(MPI_COMM_WORLD);
        Global_Allreduce(sum_vector,tmpbuf_images_ptr,N_images,MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
        cuda_copy_real_to_device(tmpbuf_images_ptr,d_sum,N_images);
        cuda_norm_respons_sumexpf(d_respons,d_sum,maxr,N_images,allocate_slice);
        total_respons = cuda_total_respons(d_respons,respons,N_images*allocate_slice);
        cuda_copy_real_to_host(respons,d_respons,allocate_slice*N_images);
        MPI_Barrier(MPI_COMM_WORLD);
        Global_Gatherv((void*)respons, recvcounts[taskid],
                       MPI_EMC_PRECISION, (void*) full_respons,
                       recvcounts,  dispal, master, MPI_COMM_WORLD);
        Global_Allreduce(&total_respons, &total_respons,1, MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);

        if(taskid ==master){
            sprintf(filename_buffer, "%s/responsabilities_%.4d.h5", conf.output_dir, iteration);
            write_2d_real_array_hdf5_transpose(filename_buffer, full_respons, N_slices, N_images);
            write_ave_respons(conf,full_respons,N_images,N_slices, iteration, average_resp);
            write_real_array(likelihood, &total_respons,1);

        }
        if(conf.isDebug == 1)
            printf("DEBUG... TOT_RESP after normalization %g %g at taskid %d\n", total_respons, total_respons,taskid);


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

            cuda_update_slices(d_images, d_slices, d_mask,
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
            cuda_normalize_model(model, d_model);
        }
        if (conf.blur_model) {
            cuda_blur_model(d_model, conf.model_side, conf.blur_model_sigma);
        }
        cuda_copy_model(model,d_model);
        cuda_copy_model(model_weight,d_model_weight);

        MPI_Barrier(MPI_COMM_WORLD);
        /*--------------model output ----------------------------------*/
        if(taskid ==master){
            write_model(conf, iteration, N_model, model,model_weight);
            write_weight(conf, iteration, N_model,model_weight);
            write_state_file_iteration(state_file, iteration);

        }
        if (iteration % N_step ==0){
            ///reload first N_reload frame
            N_acc = 0;
            while (N_acc <N_reload ){
                    N_acc += load_selected_images_by_log (conf, logF, images,masks, list,  "%.8d %ld %.4d %.1d %.4f","%s%.8d",
                                              error_lim, current_time,
                                              N_max-N_acc);
    
            }

        }
    }//END FOR ITERATION
    /* Close files that have been open throughout the aalysis. */

    close_state_file(state_file);
    fclose(timeFile);
    fclose(likelihood);
    //fclose(best_rot_file);


    if (conf.recover_scaling){
        close_scaling_file(scaling_dataset, scaling_file);
    }

    /* Reset models for a final compression with individual masks. */
    cuda_reset_model(model,d_model_updated);
    cuda_reset_model(model_weight,d_model_weight);
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
                d_model_weight,images);

    }
    cuda_copy_model(model, d_model_updated);
    cuda_copy_model(model_weight,d_model_weight);
    MPI_Barrier(MPI_COMM_WORLD);
    Global_Allreduce(model->data,model_data, N_model, MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
    Global_Allreduce(model_weight->data,model_weight_data,N_model,MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
    memcpy(model->data,model_data,sizeof(real)*N_model);
    memcpy(model_weight->data,model_weight_data,N_model*sizeof(real));
    cuda_copy_model_2_device(&d_model,model);
    cuda_copy_model_2_device(&d_model_weight,model_weight);
    cuda_divide_model_by_weight(model, d_model, d_model_weight);
    if (conf.recover_scaling){
        cuda_normalize_model(model, d_model);
    }
    cuda_copy_model(model, d_model);
    cuda_copy_model(model_weight, d_model_weight);

    /* Copy the final result to the CPU */
    write_final_model(conf, N_model, model, model_weight);
    //sprintf(filename_buffer, "%s/final_best_rotations.data", conf.output_dir);
    //FILE *final_best_rotations_file = create_file_descriptor(conf, "/final_best_rotations.data","wp");
    compute_best_rotations(full_respons, N_images, N_slices, best_rotation);
    write_best_quat(conf, conf.number_of_iterations, rotations, best_rotation, N_images);
    //fclose(final_best_rotations_file);
    MPI_Finalize();
    fclose(logF);
    return 0;
}







