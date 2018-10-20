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
    printf("%d of total %d MPI processes started!\n", taskid,ntasks);
    cuda_print_device_info();
    if (argc > 1) {
        read_configuration_file(argv[1],&conf);
    } else {
        read_configuration_file("./emc.conf",&conf);
    }
    const int N_images = conf.number_of_images;
    int slice_chunk = conf.chunk_size;
    int N_2d = conf.model_side*conf.model_side;
    int N_model = conf.model_side*conf.model_side*conf.model_side;

    sp_imatrix **masks = (sp_imatrix **)  malloc(N_images*sizeof(sp_imatrix *));
    sp_matrix **images = (sp_matrix**)malloc(conf.number_of_images*sizeof(sp_matrix *));
    for (int i =0; i<conf.number_of_images; i++){
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);
    }

    sp_imatrix * mask = read_mask(conf);
    sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_3matrix *model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
    sp_3matrix *model_weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);

    //normalize_images(images, mask, conf, central_part_radius);
   // real image_max = calcualte_image_max( mask,  images, N_images, N_2d);
    calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength, x_coordinates, y_coordinates, z_coordinates);
    model_init(conf,model, model_weight,images,mask, x_coordinates, y_coordinates, z_coordinates);
    Quaternion *rotations;
    real *weights_rotation;
    const int N_slices = read_rotations_file(conf.rotations_file, &rotations, &weights_rotation);

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


    /*----------------  GPU VAR INIT START     -------------------------------*/

    real * d_slices;
    cuda_allocate_slices(&d_slices,conf.model_side,slice_chunk);
    real * d_model;
    cuda_allocate_model(&d_model,model);
    real * d_model_updated;
    cuda_allocate_model(&d_model_updated,model);
    real * d_model_weight; //used to d_weight
    cuda_allocate_model(&d_model_weight,model_weight);


    /* List of all sampled rotations. Used in both expansion and
     compression. Does not change. */
    real *d_weights_rotation;
    cuda_allocate_real(&d_weights_rotation, N_slices);
    cuda_copy_weight_to_device(weights_rotation, d_weights_rotation, N_slices,0);
    real * d_rotations;
    cuda_allocate_rotations_chunk(&d_rotations,rotations,0,N_slices);

    real * d_x_coord;
    real * d_y_coord;
    real * d_z_coord;
    cuda_allocate_coords(&d_x_coord, &d_y_coord, &d_z_coord, x_coordinates,
                         y_coordinates,  z_coordinates);
    int * d_mask;
    cuda_allocate_mask(&d_mask,mask);

    real * d_images;
    cuda_allocate_images(&d_images,images,N_images);
   
    real *d_weight_map;
    cuda_allocate_weight_map(&d_weight_map, conf.model_side);

    real mean = cuda_model_average(d_model,N_model);
    printf("model average is %f \n",mean);
    printf("debugging: Nslices%d rot100 %f %f %f %f\n ", N_slices, rotations[0].q[0], rotations[0].q[1], rotations[0].q[2], rotations[0].q[3] );

    real* slices = (real*) malloc(sizeof(real) * N_slices*N_2d);
    cuda_get_slices(model, d_model, d_slices, d_rotations, d_x_coord, d_y_coord, d_z_coord, 0, conf.number_of_images);
    cuda_copy_real_to_host(slices,d_slices,N_images*N_2d);
    Image *write_image = sp_image_alloc(conf.model_side, conf.model_side, 1);
    for (int i_image = 0; i_image < N_slices; i_image++) {
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

        write_image->image->data[0] = sp_cinit(1000, 0.);
        sprintf(filename_buffer, "%s/image_%.4d.png", conf.output_dir, i_image);
        sp_image_write(write_image, filename_buffer, SpColormapJet|SpColormapLogScale);
    }
    sp_image_free(write_image);
    exit(0);
}



