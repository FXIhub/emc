/*
 * Author : Jing Liu@ Biophysics and TDB
 * 2016-Nov optimized kernel, branched from emc_dis.cpp
 * *
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
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <math.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <string.h>
#define  MPI_EMC_PRECISION MPI_FLOAT
using namespace std;
const real min_tol = 0;//1e-10;
const real min_tol1 = 1e-10;

const bool isAverageSliceOn = true;

struct LOG: public thrust::unary_function<real, real>
{
    __host__ __device__  real  operator()(real x)
    {
        return x>min_tol? logf(x) :0;
    }
};
struct INV: public thrust::unary_function<real, real>
{
    __host__ __device__  real  operator()(real x)
    {
        return x>min_tol? 1/x :0;
    }
};

struct INV2: public thrust::unary_function<real, real>
{
    __host__ __device__  real  operator()(real x)
    {
        return x>min_tol1? 1/x :1;
    }
};

struct IsZeros: public thrust::unary_function<real, real>
{
    __host__ __device__  real  operator()(real x)
    {
        return x>min_tol?1:0;
    }
};

struct divs : public thrust::binary_function<real,real, real>
{
    __host__ __device__ real operator()(const real x, const real y) const {return y>min_tol?x/y:0;}
};

int main(int argc, char *argv[]){
    Configuration conf;

    /*-----------------------------------------------------------Do Master Node Initial Job start------------------------------------------------*/
    cout<<"Init MPI...";
    MPI_Init(&argc, &argv);
    int taskid, ntasks;
    int master = 0;
    unsigned long int timeB = 0;
    unsigned long int timeS = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
    printf("%d of total %d MPI processes started!\n", taskid,ntasks);
    if(taskid == master) cuda_print_device_info();
    if (argc > 1) {
        read_configuration_file(argv[1],&conf);
    } else {
        read_configuration_file("./emc.conf",&conf);
    }
    real  *model_data =(real*) malloc(sizeof(real)*conf.model_side*conf.model_side*conf.model_side);
    real  *model_weight_data =(real*) malloc(sizeof(real)*conf.model_side*conf.model_side*conf.model_side);
    sp_imatrix **masks = (sp_imatrix **)  malloc(conf.number_of_images*sizeof(sp_imatrix *));
    sp_matrix **images = read_images(conf,masks);
    sp_imatrix * mask = read_mask(conf);
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
    if(conf.normalize_images == 1)
        normalize_images(images, mask, conf, central_part_radius);
    real image_max = calcualte_image_max( mask,  images, N_images, N_2d);
    cout << "image max is "<< image_max<<endl;
    calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength, x_coordinates, y_coordinates, z_coordinates);
    model_init(conf,model, model_weight,images,mask, x_coordinates, y_coordinates, z_coordinates);
    Quaternion *rotations;
    real *weights_rotation;
    const int N_slices = read_rotations_file(conf.rotations_file, &rotations, &weights_rotation);
    //cout << "N_slices is " << N_slices <<endl;
    //const int N_slices = generate_rotation_list(8,&rotations,&weights_rotation);
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
    
    /*------------------------------task devision of respons  MPI BUFFER----------------------------*/
    
    
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
    real *radial_fit = (real*) malloc(round(conf.model_side/2.0)*sizeof(real));
    real *radial_fit_weight = (real*) malloc(round(conf.model_side/2.0)*sizeof(real));
    real *d_radial_fit;
    real *d_radial_fit_weight;
    cuda_allocate_real(&d_radial_fit, round(conf.model_side/2.0));
    cuda_allocate_real(&d_radial_fit_weight,round(conf.model_side/2.0));
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
    FILE *timeFileStep = create_file_descriptor(conf, "/exeTimeByStep.data", "wp");

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
    /*-----------------------------------------------------------Do local init END-----------------------------------------------------------*/
    
    
    
    
    /*------------------------------for cublas optimized kernel------------------------------------*/
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cublasHandle_t hd;
    cublasCreate(&hd);
    thrust::device_vector<real> thrust_responstmp(N_images*allocate_slice,0);
    real* d_respons_tmp = thrust::raw_pointer_cast(&thrust_responstmp[0]);
    thrust::device_vector<real> thrust_onesN2d(N_2d,1);
    thrust::device_vector<real> thrust_onesNimges(N_images,1);

    thrust::device_vector<real> thrust_onesN2dslicechunck(N_2d*slice_chunk ,1);
    thrust::device_vector<real> thrust_sum_pix_of_slices(slice_chunk ,0);
    thrust::device_vector<real> thrust_sum_pix_of_image(N_images ,0);
    thrust::device_vector<real> thrust_sum_img_of_respons(allocate_slice ,0);
    thrust::device_vector<real> thrust_sum_img_of_respons_inv(allocate_slice ,0);

    real* d_onesN2d = thrust::raw_pointer_cast(&thrust_onesN2d[0]);
    real* d_onesNimages = thrust::raw_pointer_cast(&thrust_onesNimges[0]);

    real* d_onesN2dslicechunck = thrust::raw_pointer_cast(&thrust_onesN2dslicechunck[0]);
    real* d_sum_img_of_respons = thrust::raw_pointer_cast(&thrust_sum_img_of_respons[0]);
    real* d_sum_img_of_respons_inv = thrust::raw_pointer_cast(&thrust_sum_img_of_respons_inv[0]);


    real* d_sum_pix_of_slices = thrust::raw_pointer_cast(&thrust_sum_pix_of_slices[0]);
    real* d_sum_pix_of_image = thrust::raw_pointer_cast(&thrust_sum_pix_of_image[0]);
    thrust:: device_vector<real> thrust_log_phi (N_images*allocate_slice ,1);
    //thrust:: device_vector<real> thrust_log_slices (N_2d*slice_chunk ,1);
    real* d_log_phi = thrust::raw_pointer_cast(&thrust_log_phi[0]);
    //real* d_log_slices = thrust::raw_pointer_cast(&thrust_log_slices[0]);

    thrust::device_ptr<real> d_Respons_begin(d_respons);
    thrust::device_ptr<real> d_Respons_end = d_Respons_begin +  allocate_slice*N_images;
    thrust::device_ptr<real> d_slice_begin(d_slices);
    thrust::device_ptr<real> d_slice_end = d_slice_begin +  slice_chunk*N_2d;
    thrust::device_ptr<real> d_scaling_begin(d_scaling);

    //thrust::device_vector<real> thrust_onesN2dNimages(N_2d*N_images,1);
    //real* d_onesN2dNimages = thrust::raw_pointer_cast(&thrust_onesN2dNimages[0]);


    const real one = 1.0;
    const real mone = -1.0;
    const real zero = 0;
    cublasStatus_t st;

    double exeTimeStep =0;

    /*------------------------------for cublas optimized kernel------------------------------------*/
    
    /*------------------------------------ EMC Iterations start ----------------------------------------------------------*/
    for (int iteration = start_iteration; iteration < conf.number_of_iterations; iteration++) {

        /*---------------------- reset local variable-------------------------------*/
        if(taskid == master){
            if (iteration == start_iteration ){
                timeB = gettimenow();
                timeS = timeB;
            }
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
        //cout <<conf.diff<<endl<<endl;
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
        cuda_set_to_zero(d_fit,N_images);
        cuda_set_to_zero(d_radial_fit,conf.model_side/2.0);
        cuda_set_to_zero(d_radial_fit_weight,conf.model_side/2.0);
        

        /*---------------------- Calculate scaling start , opt checked right-------------------------*/
        //printf(" cal scaling! \n\n");
        // only poissonion case is considered
        //sum_i K_ik
        timeS = gettimenow();

        if(conf.isDebug==1){
            //cuda_apply_single_mask(d_images,d_mask,N_2d,N_images);//mask!=0
            cuda_reset_real(d_sum_pix_of_image, N_images);
            //y = α op ( A ) x + β y
            st = cublasSgemv(hd,CUBLAS_OP_T, N_2d, N_images,
                             &one,d_images, N_2d, d_onesN2d, 1, &zero,d_sum_pix_of_image, 1);
            // cout << "Scaling  sum_i K_ij'" <<thrust_sum_pix_of_image[0] <<" "<< thrust_sum_pix_of_image[100]<<endl;

        }

        if (conf.recover_scaling) {
            cuda_reset_real(d_scaling, N_images*allocate_slice);
            printf("reconver scaling \n\n");
            for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
                if (slice_start + slice_chunk >= slice_end) {
                    current_chunk = slice_end - slice_start;
                } else {
                    current_chunk = slice_chunk;
                }
                int current_start = slice_start- slice_backup;
                cuda_reset_real(d_slices, N_2d*slice_chunk);
                cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord,
                                current_start   , current_chunk);

                if(conf.isDebug == 1)
                {
                    cuda_reset_real(d_sum_pix_of_slices, slice_chunk);
                    //sum_i W_ij'
                    st = cublasSgemv(hd,CUBLAS_OP_T,  N_2d, slice_chunk,
                                     &one,d_slices, N_2d, d_onesN2d , 1, &zero,d_sum_pix_of_slices, 1);
                    //cout << "cublasSgemv in sum_i Wij" <<st<<endl;

                    thrust::transform(thrust_sum_pix_of_slices.begin(), thrust_sum_pix_of_slices.end(),
                                      thrust_sum_pix_of_slices.begin(), INV());
                    //A = α x y T + A if ger(),geru() is called α x y H + A if gerc() is called
                    //where A is a m × n matrix stored in column-major format, x and y are vectors, and α is a scalar.
                    st = cublasSger(hd, N_images,  slice_chunk, &one, d_sum_pix_of_image ,
                                    1,  d_sum_pix_of_slices , 1,  &d_scaling[current_start*N_images], N_images);
                    //cout << "cublasSger in sum_i Kik / sum_i W_ij" <<st<<endl;
                }
                else{
                    cuda_update_scaling_full(d_images, d_slices, d_mask, d_scaling, d_weight_map,
                                             N_2d, N_images, current_start, current_chunk,diff_type(conf.diff));
                }
            }
            if(taskid ==master){
                exeTimeStep=  update_time(timeS, gettimenow());
                write_time_by_step(timeFileStep,  exeTimeStep,  iteration,"Scaling");
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
                    //if(conf.isDebug == 1)
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
        cuda_set_to_zero(d_respons_tmp,N_images*allocate_slice);
        if(taskid ==master)
            timeS = gettimenow();

        /* In this loop through the chunks the responsabilities are updated. */
        if(conf.isDebug==1 ){
            cudaMemcpy (d_log_phi, d_scaling,sizeof(real)*N_images*allocate_slice,cudaMemcpyDeviceToDevice);
            //log(phi_kj')
            thrust::transform(thrust_log_phi.begin(), thrust_log_phi.end(), thrust_log_phi.begin(), LOG());
            if( !isAverageSliceOn){
                st = cublasSdgmm(hd, CUBLAS_SIDE_LEFT
                                 , N_images, allocate_slice,d_log_phi, N_images,
                                 d_sum_pix_of_image, 1, &d_respons_tmp[0], N_images);
            }
            //cout << "cublasSdgmm in  log(phi_kj') *sum_i K_ik of Respons" <<st<<endl;

        }
        for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= slice_end) {
                current_chunk = slice_end - slice_start;
            } else {
                current_chunk = slice_chunk;
            }
            int current_start = slice_start- slice_backup;
            //cout<<current_start<<endl<<endl;
            cuda_reset_real(d_slices, N_2d*slice_chunk);
            cuda_get_slices(model,d_model,d_slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,
                            current_start,current_chunk);
            cuda_apply_single_mask_zeros(d_images,d_mask,N_2d,N_images);//mask!=0
            cuda_apply_single_mask_zeros(d_slices,d_mask,N_2d,slice_chunk);//mask!=0

            if(conf.isDebug==1){
                //cuda_apply_single_mask_zeros(d_slices,d_mask,N_2d,slice_chunk);//mask!=0
                //cuda_apply_single_mask_zeros(d_images,d_mask,N_2d,N_images);//mask!=0
                if (isAverageSliceOn){
                    // Wij>0
                    cuda_reset_real(d_onesN2dslicechunck, slice_chunk *N_2d);
                    thrust::transform(d_slice_begin, d_slice_end, thrust_onesN2dslicechunck.begin(), IsZeros());
                    //K_ik*B_ij
                    st = cublasSgemm(hd,  CUBLAS_OP_T,  CUBLAS_OP_N, N_images,slice_chunk, N_2d,  &one,
                                     d_images,N_2d, d_onesN2dslicechunck, N_2d, &zero,
                                     &d_respons_tmp[current_start*N_images] , N_images);
                }

                //sum_i Wij'
                cuda_reset_real(d_sum_pix_of_slices,slice_chunk);
                st = cublasSgemv(hd,CUBLAS_OP_T,  N_2d, slice_chunk,
                                 &one,d_slices, N_2d, d_onesN2d , 1, &zero,d_sum_pix_of_slices, 1);
                // cout << "cublasSgemv in sum_i W_ij of Respons" <<st <<endl;
                //cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const real *A, int lda,
                //const real *x, int incx, real *C, int ldc)
                // phi_kj' * sum_i W_ij'

                st = cublasSdgmm(hd, CUBLAS_SIDE_RIGHT,
                                 N_images,slice_chunk, &d_scaling[current_start*N_images],N_images,
                        d_sum_pix_of_slices, 1, &d_respons[current_start*N_images], N_images);

                //cout << "cublasSdgmm in  phi_j'k * sum_ij' of Respons checked right" <<st<<endl;
                // log(W_ij')
                thrust::transform(d_slice_begin, d_slice_end, d_slice_begin, LOG());
                //cublasStatus_t cublasSgemm(cublasHandle_t handle,  transa,  transb, int m, int n, int k,  *alpha,
                //const float *A, int lda, const float *B, int ldb, const float *beta,float  *C, int ldc)
                //op ( A ) m × k , op ( B ) k × n and C m × n
                //C = α op ( A ) op ( B ) + β C

                st = cublasSgemm(hd,  CUBLAS_OP_T,  CUBLAS_OP_N, N_images,slice_chunk, N_2d,  &one,
                                 d_images,N_2d, d_slices, N_2d, &mone,
                                 &d_respons[current_start*N_images] , N_images);
                //cout << "cublasSgemm in  log(W_ij'') *K_ik of Respons" <<st<<endl;
            }
            else{
                cuda_calculate_responsabilities(d_slices, d_images, d_mask, d_weight_map,
                                                sigma, d_scaling, d_respons, d_weights_rotation,
                                                N_2d, N_images, current_start,
                                                current_chunk,  diff_type(conf.diff));
            }
            
        }
        if(conf.isDebug==1){
            if (isAverageSliceOn){
                thrust::transform( thrust_responstmp.begin(), thrust_responstmp.end(),
                                   thrust_log_phi.begin(),thrust_responstmp.begin(),thrust::multiplies<real>());
            }
            thrust::transform( thrust_responstmp.begin(), thrust_responstmp.end(),
                               d_Respons_begin,d_Respons_begin,thrust::plus<real>());


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
        if(taskid ==master){
            exeTimeStep =  update_time(timeS, gettimenow());
            write_time_by_step(timeFileStep,  exeTimeStep,  iteration,"Respons");
        }
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
        printf("DEBUG... TOT_RESP after normalization %g %g at taskid %d\n", total_respons, total_respons,taskid);
        MPI_Barrier(MPI_COMM_WORLD);
        /*-----------------------------end respons normalization (distribution ) done-------------------------------*/

        /*-----------------------------start update slice and insert slice -------------------------------*/

        /* start update model */
        cuda_reset_model(model,d_model_updated);
        cuda_reset_model(model_weight,d_model_weight);
        //cout <<"UPDATE MODEL!" <<endl;
        if(taskid ==master)
            timeS = gettimenow();
        if(conf.recover_scaling){
            //sum_k P_jk
            cuda_copy_real(d_respons_tmp,d_respons, allocate_slice*N_images);

            cuda_reset_real(d_sum_img_of_respons,N_images);
            st = cublasSgemv(hd,CUBLAS_OP_T,  N_images, allocate_slice,
                             &one,d_respons, N_images, d_onesNimages ,
                             1, &zero,d_sum_img_of_respons, 1);

            //1/total_respons
            thrust::transform(thrust_sum_img_of_respons.begin(), thrust_sum_img_of_respons.end(),
                              thrust_sum_img_of_respons_inv.begin(),INV2());
            // 1/total_respons
            //thrust::transform(thrust_sum_img_of_respons.begin(), thrust_sum_img_of_respons.end(),
            //                  thrust_sum_img_of_respons.begin(),INV3());
            //P_jk ./phi_jk
            thrust::transform( d_Respons_begin, d_Respons_end,
                               d_scaling_begin,thrust_responstmp.begin(),divs());
        }

        for (slice_start = slice_backup; slice_start < slice_end; slice_start += slice_chunk) {
            if (slice_start + slice_chunk >= slice_end) {
                current_chunk = slice_end - slice_start;
            } else {
                current_chunk = slice_chunk;
            }
            int current_start = slice_start- slice_backup;
            if(conf.isDebug == 1){
                //calculate Wij then call insert slices
                //cublasStatus_t cublasSgemm(cublasHandle_t handle,  transa,  transb, int m, int n, int k,  *alpha,
                //const float *A, int lda, const float *B, int ldb, const float *beta,float  *C, int ldc)
                //op ( A ) m × k , op ( B ) k × n and C m × n
                //C = α op ( A ) op ( B ) + β C
                cuda_reset_real(d_slices, N_2d*slice_chunk);

                st = cublasSgemm(hd,  CUBLAS_OP_N,  CUBLAS_OP_N, N_2d,slice_chunk, N_images,  &one,
                                 d_images,N_2d,&d_respons_tmp[current_start*N_images], N_images, &zero,
                        d_slices, N_2d);
                cuda_apply_single_mask_zeros(d_slices,d_mask,N_2d,slice_chunk);
                
                // Wij/ phi_j(total_respons)
                //cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const real *A, int lda,
                //const real *x, int incx, real *C, int ldc)


                st = cublasSdgmm(hd, CUBLAS_SIDE_RIGHT,
                                 N_2d,slice_chunk, d_slices,N_2d,
                                 &d_sum_img_of_respons_inv[current_start], 1, d_slices, N_2d);

                cuda_insert_slices(d_images, d_slices, d_mask,
                                   d_respons, d_scaling, &d_sum_img_of_respons[current_start],d_active_images,
                                   N_images, current_start, current_chunk, N_2d,
                                   model,d_model_updated, d_x_coord, d_y_coord,
                                   d_z_coord, &d_rotations[current_start*4],
                        d_model_weight,images);

            }else{
                cuda_update_slices(d_images, d_slices, d_mask,
                                   d_respons, d_scaling, d_active_images,
                                   N_images, current_start, current_chunk, N_2d,
                                   model,d_model_updated, d_x_coord, d_y_coord,
                                   d_z_coord, &d_rotations[current_start*4],
                        d_model_weight,images);
            }

        }
        d_model_tmp = d_model_updated;
        d_model_updated = d_model;
        d_model = d_model_tmp;

        if(taskid ==master){
            exeTimeStep =  update_time(timeS, gettimenow());
            write_time_by_step(timeFileStep,  exeTimeStep,  iteration,"slices");
        }
        /*-----------------------------end update slice and insert slice -------------------------------*/

        
        // collect model and model_weight back to the master node
        cuda_copy_model(model,d_model);
        cuda_copy_model(model_weight,d_model_weight);
        MPI_Barrier(MPI_COMM_WORLD);
        Global_Allreduce(model->data,model_data, N_model, MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
        Global_Allreduce(model_weight->data,model_weight_data,N_model,MPI_EMC_PRECISION, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        memcpy(model->data,model_data,sizeof(real)*N_model);
        memcpy(model_weight->data,model_weight_data,N_model*sizeof(real));

        cuda_copy_model_2_device(&d_model,model);
        cuda_copy_model_2_device(&d_model_weight,model_weight);
        sum = cuda_model_average(d_model,N_model);
        printf("model average is %g before dividing modle_weight at %d\n", sum, taskid);
        cuda_divide_model_by_weight(model, d_model, d_model_weight);

        sum = cuda_model_average(d_model,N_model);
        printf("model average is %g before normalize_model at %d\n", sum, taskid);

        if (conf.recover_scaling){
            cuda_normalize_model(model, d_model);
        }
        sum = cuda_model_average(d_model,N_model);
        printf("model average is %g after normalize_model at %d\n", sum, taskid);

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
    }//END FOR ITERATION
    /* Close files that have been open throughout the aalysis. */
    
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
    return 0;
}





//

