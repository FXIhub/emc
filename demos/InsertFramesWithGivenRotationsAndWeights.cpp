
/*
 * Author : Jing Liu@ Biophysics and TDB
 * 2018-03-13
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

int main(int argc, char *argv[]){
    Configuration conf;
    /*-----------------------------------------------------------Do Master Node Initial Job start------------------------------------------------*/

    cuda_print_device_info();
    if (argc > 1) {
        read_configuration_file(argv[1],&conf);
    } else {
        read_configuration_file("./insert.conf",&conf);
    }
    Quaternion *rotations;
    real *weights_rotation;
    int N_slices = read_rotations_file(conf.rotations_file, &rotations, &weights_rotation);

    const int N_images = conf.number_of_images;
    const int N_2d = conf.model_side*conf.model_side;
    const int N_model = conf.model_side*conf.model_side*conf.model_side;
   // int N_slices = N_images;

    real *respons = (real*) malloc(N_slices*N_images*sizeof(real));
    //real *scaling = (real*) malloc(N_slices*N_images*sizeof(real));

    for (int i =0; i<N_slices; i++){
        for (int j = 0;j<N_images; j++){
            if (i == j)        respons[i+j*N_images] = 1;
            else respons[i+j*N_images] = 0;
        }
    }


    sp_imatrix **masks = (sp_imatrix **)  malloc(N_images*sizeof(sp_imatrix *));
    sp_matrix **images = read_images(conf,masks);
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

    int* active_images = (int*) malloc(N_images*sizeof(int));
    real* slice_respons = (real*) malloc(N_slices*sizeof(real));
    for (int i = 0; i<N_images; i++){
        active_images[i]= 1;
        slice_respons[i] = 1.0;
    }

    int * d_mask;
    cuda_allocate_mask(&d_mask,mask);

    real * d_images;
    cuda_allocate_images(&d_images,images,N_images);
    cuda_apply_single_mask(d_images, d_mask, N_2d, N_images);

    real * d_respons;
    cuda_allocate_real(&d_respons,N_images*N_slices);
    cuda_copy_real_to_device(respons,d_respons,N_images*N_slices);

    //scaling
    real * d_scaling;
    cuda_allocate_real(&d_scaling,N_slices*N_images);
    cuda_copy_real_to_device(respons,d_scaling, N_slices*N_images);

    int* d_active_images ;
    cuda_allocate_int(&d_active_images, N_images);
    cuda_copy_int_to_device(active_images,d_active_images,N_images);

    real* d_model;
    cuda_allocate_model(&d_model, model);

    real * d_x_coord;
    real * d_y_coord;
    real * d_z_coord;
    cuda_allocate_coords(&d_x_coord, &d_y_coord, &d_z_coord, x_coordinates,
                         y_coordinates,  z_coordinates);

    real *d_weights_rotation;
    cuda_allocate_real(&d_weights_rotation, N_slices);
    cuda_copy_weight_to_device(weights_rotation, d_weights_rotation, N_slices,0);
    real * d_rotations;
    cuda_allocate_rotations(&d_rotations,rotations,N_slices);
    cuda_copy_rotations_chunk(&d_rotations, rotations,0,N_slices);

    real * d_model_weight; //used to d_weight
    cuda_allocate_model(&d_model_weight,model_weight);

    real* d_slices_total_respons;
    cuda_allocate_real(&d_slices_total_respons,N_slices);
    cuda_copy_real_to_device(slice_respons,d_slices_total_respons,N_slices);

    cuda_insert_slices(d_images, d_images, d_mask,
                       d_respons, d_scaling, d_slices_total_respons,d_active_images,
                       N_images, 0, N_slices, N_2d,
                       model,d_model, d_x_coord, d_y_coord,
                       d_z_coord, &d_rotations[0],
            d_model_weight,images);

    cuda_divide_model_by_weight(model, d_model, d_model_weight);
    cuda_copy_model(model,d_model);
    cuda_copy_model(model_weight,d_model_weight);

    /*--------------model output ----------------------------------*/
    write_model(conf, 0, N_model, model,model_weight);
    write_weight(conf, 0, N_model,model_weight);

}
