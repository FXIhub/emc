/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include <IOHDF5.h>
/* Writes any real type array in hdf5 format*/

void write_1d_real_array_hdf5(char *filename, real *array, int index1_max) {
    hid_t file_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id;
    hid_t dataset_id;
    hsize_t dim[1]; dim[0] = index1_max;
    
    space_id = H5Screate_simple(1, dim, NULL);
    dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
}

/* Writes any int type array in hdf5 format. */
void write_1d_int_array_hdf5(char *filename, int *array, int index1_max) {
    hid_t file_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id;
    hid_t dataset_id;
    hsize_t dim[1]; dim[0] = index1_max;
    
    space_id = H5Screate_simple(1, dim, NULL);
    dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
}


/* Initialize a file that will contain the best scaling for each diffraction
 pattern for every iteratino. */
hid_t init_scaling_file(char *filename, int N_images, hid_t *file_id) {
    *file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {printf("Error creating file\n");}
    
    const int init_stack_size = 10;
    hsize_t dims[2]; dims[0] = init_stack_size; dims[1] = N_images;
    hsize_t maxdims[2]; maxdims[0] = H5S_UNLIMITED; maxdims[1] = N_images;
    
    hid_t dataspace = H5Screate_simple(2, dims, maxdims);
    if (dataspace < 0) error_exit_with_message("Error creating scaling dataspace\n");
    
    hid_t cparms = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(cparms, 2, dims);
    
    hid_t dataset = H5Dcreate(*file_id, "scaling", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, cparms, H5P_DEFAULT);
    if (dataset < 0) error_exit_with_message("Error creating scaling dataset\n");
    H5Pset_chunk_cache(H5Dget_access_plist(dataset), H5D_CHUNK_CACHE_NSLOTS_DEFAULT, N_images, 1);
    
    H5Sclose(dataspace);
    H5Pclose(cparms);
    return dataset;
}

/* Write the best scaling for each diffraction pattern to an already open
 hdf5 file. */
void write_scaling_to_file(const hid_t dataset, const int iteration, float *scaling, float *respons, int N_slices) {
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0) error_exit_with_message("Can not create dataset when writing scaling.\n");
    hsize_t block[2];
    hsize_t mdims[2];
    H5Sget_simple_extent_dims(dataspace, block, mdims);
    
    /* Check if stack has enough space, otherwhise enlarge. */
    if (block[0] <= iteration) {
        while(block[0] <= iteration) {
            block[0] *= 2;
        }
        H5Dset_extent(dataset, block);
        H5Sclose(dataspace);
        dataspace = H5Dget_space(dataset);
        if (dataspace<0) error_exit_with_message("Error enlarging dataspace in scaling\n");
    }
    
    /* Now that we know the extent, find the best responsability and create an
     array with the scaling. */
    int N_images = block[1];
    float *data = (float *) malloc(N_images*sizeof(float));
    int best_index;
    float best_resp;
    for (int i_image = 0; i_image < N_images; i_image++) {
        best_resp = respons[i_image];
        best_index = 0;
        for (int i_slice = 1; i_slice < N_slices; i_slice++) {
            if (respons[i_slice*N_images + i_image] > best_resp) {
                best_resp = respons[i_slice*N_images + i_image];
                best_index = i_slice;
            }
        }
        data[i_image] = scaling[best_index*N_images + i_image];
    }
    
    /* Get the hdf5 hyperslab for the current iteration */
    block[0] = 1;
    hid_t memspace = H5Screate_simple(2, block, NULL);
    if (memspace < 0) error_exit_with_message("Error creating memspace when writing scaling\n");
    hsize_t offset[2] = {iteration, 0};
    hsize_t stride[2] = {1, 1};
    hsize_t count[2] = {1, 1};
    hid_t hyperslab = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, stride, count, block);
    if (hyperslab < 0) error_exit_with_message("Error selecting hyperslab in scaling\n");
    hid_t w = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data);
    if (w < 0) error_exit_with_message("Error writing scaling\n");
    free(data);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Fflush(dataset, H5F_SCOPE_GLOBAL);
}

/* Close scaling file. Call this after the last call to
 write_scaling_to_file. */
void close_scaling_file(hid_t dataset, hid_t file) {
    H5Sclose(dataset);
    H5Sclose(file);
}

/* Writes any real type 2D array in hdf5 format. */
void write_2d_real_array_hdf5(char *filename, real *array, int index1_max, int index2_max) {
    hid_t file_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id;
    hid_t dataset_id;
    hsize_t dim[2]; dim[0] = index1_max; dim[1] = index2_max;
    
    space_id = H5Screate_simple(2, dim, NULL);
    dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
}

/* Writes any real type 2D array in hdf5 format. Transposes the
 array before writing. */
void write_2d_real_array_hdf5_transpose(char *filename, real *array, int index1_max, int index2_max) {
    
    real *array_trans = (real*) malloc(index1_max*index2_max*sizeof(real));
    for (int i1 = 0; i1 < index1_max; i1++) {
        for (int i2 = 0; i2 < index2_max; i2++) {
            array_trans[i2*index1_max+i1] = array[i1*index2_max+i2];
        }
    }
    hid_t file_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id;
    hid_t dataset_id;
    hsize_t dim[2]; dim[0] = index2_max; dim[1] = index1_max;
    
    space_id = H5Screate_simple(2, dim, NULL);
    dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array_trans);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
    free(array_trans);
}

/* Writes any real type 3D array in hdf5 format. */
void write_3d_array_hdf5(char *filename, real *array, int index1_max, int index2_max, int index3_max) {
    hid_t file_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id;
    hid_t dataset_id;
    hsize_t dim[3];
    dim[0] = index1_max; dim[1] = index2_max; dim[2] = index3_max;
    
    space_id = H5Screate_simple(3, dim, NULL);
    dataset_id = H5Dcreate1(file_id, "/data", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
}



sp_matrix **read_images_cxi(const char *filename, const char *image_identifier, const char *mask_identifier,
                            const int number_of_images, const int image_side, const int binning_factor,
                            sp_imatrix **list_of_masks) {
    int status;
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) error_exit_with_message("Problem reading file %s", filename);
    hid_t dataset = H5Dopen1(file, image_identifier);
    if (dataset < 0) error_exit_with_message("Problem reading dataset %s in file %s", image_identifier, filename);
    
    hsize_t dims[3];
    hid_t file_dataspace = H5Dget_space(dataset);
    H5Sget_simple_extent_dims(file_dataspace, dims, NULL);
    if (number_of_images > dims[0]) error_exit_with_message("Dataset in %s does not contain %d images", filename, number_of_images);
    
    hsize_t hyperslab_start[3] = {0, 0, 0};
    hsize_t read_dims[3] = {number_of_images, dims[1], dims[2]};
    const int total_size = read_dims[0]*read_dims[1]*read_dims[2];
    status = H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, hyperslab_start, NULL, read_dims, NULL);
    if (status < 0) error_exit_with_message("error selecting hyperslab in file %s", filename);
    
    hid_t data_dataspace = H5Screate_simple(3, read_dims, NULL);
    
    printf("read images\n");
    real *raw_image_data = (real*)malloc(total_size*sizeof(real));
    status = H5Dread(dataset, H5T_NATIVE_FLOAT, data_dataspace, file_dataspace, H5P_DEFAULT, raw_image_data);
    if (status < 0) error_exit_with_message("error reading data in file %s", filename);
    H5Dclose(dataset);
    
    dataset = H5Dopen1(file, mask_identifier);
    if (dataset < 0) error_exit_with_message("Problem reading dataset %s in file %s", mask_identifier, filename);
    
    printf("read mask\n");
    int *raw_mask_data = (int*)malloc(total_size*sizeof(int));
    status = H5Dread(dataset, H5T_NATIVE_INT, data_dataspace, file_dataspace, H5P_DEFAULT, raw_mask_data);
    H5Dclose(dataset);
    
    H5Sclose(data_dataspace);
    H5Sclose(file_dataspace);
    
    H5Fclose(file);
    
    /* Any pixels with values below zero are set to zero */
    for (int index = 0; index < total_size; index++) {
        if (raw_image_data[index] < 0.) {
            raw_image_data[index] = 0.;
        }
    }
    
    sp_matrix **list_of_images = (sp_matrix**)malloc(number_of_images*sizeof(sp_matrix *));
    real pixel_sum, pixel_this;
    int mask_sum, mask_this;
    int transformed_x, transformed_y;
    int pixel_index;
    for (int image_index = 0; image_index < number_of_images; image_index++) {
        /* Allocate return arrays */
        list_of_images[image_index] = sp_matrix_alloc(image_side, image_side);
        list_of_masks[image_index] = sp_imatrix_alloc(image_side, image_side);
        
        /* Downsample images and masks by nested loops of all
         downsampled pixels and then all subpixels. */
        for (int x = 0; x < image_side; x++) {
            for (int y = 0; y < image_side; y++) {
                pixel_sum = 0.0;
                mask_sum = 0;
                /* Step through all sub-pixels and add up values */
                for (int xb = 0; xb < binning_factor; xb++) {
                    for (int yb = 0; yb < binning_factor; yb++) {
                        transformed_x = dims[2]/2 - (image_side/2)*binning_factor + x*binning_factor + xb;
                        transformed_y = dims[1]/2 - (image_side/2)*binning_factor + y*binning_factor + yb;
                        pixel_index = image_index*dims[1]*dims[2] + transformed_y*dims[2] + transformed_x;
                        if (transformed_x >= 0 && transformed_x < dims[2] &&
                                transformed_y >= 0 && transformed_y < dims[1]) {
                            pixel_this = raw_image_data[pixel_index];
                            mask_this = raw_mask_data[pixel_index];
                        } else {
                            pixel_this = 0.;
                            mask_this = 0;
                        }
                        if (mask_this > 0) {
                            pixel_sum += pixel_this;
                            mask_sum += 1;
                        }
                    }
                }
                /* As long as there were at least one subpixel contributin to the
                 pixel (that is that was not masked out) we include the data and
                 don't mask out the pixel. */
                if (mask_sum > 0) {
                    sp_matrix_set(list_of_images[image_index], x, y, pixel_sum/(real)mask_sum);
                    sp_imatrix_set(list_of_masks[image_index], x, y, 1);
                } else {
                    sp_matrix_set(list_of_images[image_index], x, y, 0.);
                    sp_imatrix_set(list_of_masks[image_index], x, y, 0);
                }
            }
        }
    }
    return list_of_images;
}


sp_matrix **read_images_by_list(Configuration conf, sp_imatrix **masks, int* lst)
{
    sp_matrix **images = (sp_matrix**)malloc(conf.number_of_images*sizeof(sp_matrix *));
    //masks = malloc(conf.number_of_images*sizeof(sp_imatrix *));
    Image *img;
    real *intensities = (real*)malloc(conf.number_of_images*sizeof(real));
    char filename_buffer[PATH_MAX];

    for (int i = 0; i < conf.number_of_images; i++) {
        intensities[i] = 1.0;
    }

    for (int i = 0; i < conf.number_of_images; i++) {
        sprintf(filename_buffer,"%s%.4d.h5", conf.image_prefix, lst[i]);
        img = sp_image_read(filename_buffer,0);

        /* Allocate return arrays */
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);

        /* The algorithm can't handle negative data so if we have negative
         values they are simply set to 0. */
        for (int pixel_i = 0; pixel_i < sp_image_size(img); pixel_i++) {
            if (sp_real(img->image->data[pixel_i]) < 0.) {
                sp_real(img->image->data[pixel_i]) = 0.;
            }
        }

        /* Downsample images and masks by nested loops of all
         downsampled pixels and then all subpixels. */
        real pixel_sum, pixel_this;
        int mask_sum, mask_this;
        int transformed_x, transformed_y;
        for (int x = 0; x < conf.model_side; x++) {
            for (int y = 0; y < conf.model_side; y++) {
                pixel_sum = 0.0;
                mask_sum = 0;
                /* Step through all sub-pixels and add up values */
                for (int xb = 0; xb < conf.image_binning; xb++) {
                    for (int yb = 0; yb < conf.image_binning; yb++) {
                        transformed_x = sp_image_x(img)/2 - (conf.model_side/2)*conf.image_binning + x*conf.image_binning + xb;
                        transformed_y = sp_image_y(img)/2 - (conf.model_side/2)*conf.image_binning + y*conf.image_binning + yb;
                        if (transformed_x >= 0 && transformed_x < sp_image_x(img) &&
                                transformed_y >= 0 && transformed_y < sp_image_y(img)) {
                            pixel_this = sp_cabs(sp_image_get(img, transformed_x, transformed_y, 0)) ;
                            mask_this = sp_image_mask_get(img,  transformed_x, transformed_y, 0);
                        } else {
                            pixel_this = 0.;
                            mask_this = 0;
                        }
                        if (mask_this > 0) {
                            pixel_sum += pixel_this;
                            mask_sum += 1;
                        }
                    }
                }
                /* As long as there were at least one subpixel contributin to the
                 pixel (that is that was not masked out) we include the data and
                 don't mask out the pixel. */
                if (mask_sum > 0) {
                 //for tmp use
                    sp_matrix_set(images[i],x,y,pixel_sum/(real)mask_sum);
                    sp_imatrix_set(masks[i],x,y,1);
                } else {
                    sp_matrix_set(images[i],x,y,0.);
                    sp_imatrix_set(masks[i],x,y,0);
                }
            }
        }
        sp_image_free(img);
    }
    return images;
}


/* Read images in spimage format and read the individual masks. The
 masks pointer should be allocated before calling, it is not
 allocated by this function. */
sp_matrix **read_images(Configuration conf, sp_imatrix **masks)
{
    sp_matrix **images = (sp_matrix**)malloc(conf.number_of_images*sizeof(sp_matrix *));
    //masks = malloc(conf.number_of_images*sizeof(sp_imatrix *));
    Image *img;
    real *intensities = (real*)malloc(conf.number_of_images*sizeof(real));
    char filename_buffer[PATH_MAX];
    
    for (int i = 0; i < conf.number_of_images; i++) {
        intensities[i] = 1.0;
    }
    
    for (int i = 0; i < conf.number_of_images; i++) {
        sprintf(filename_buffer,"%s%.4d.h5", conf.image_prefix, i);
        img = sp_image_read(filename_buffer,0);
        
        /* Allocate return arrays */
        images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
        masks[i] = sp_imatrix_alloc(conf.model_side,conf.model_side);
        
        /* The algorithm can't handle negative data so if we have negative
         values they are simply set to 0. */
        for (int pixel_i = 0; pixel_i < sp_image_size(img); pixel_i++) {
            if (sp_real(img->image->data[pixel_i]) < 0.) {
                sp_real(img->image->data[pixel_i]) = 0.;
            }
        }
        
        /* Downsample images and masks by nested loops of all
         downsampled pixels and then all subpixels. */
        real pixel_sum, pixel_this;
        int mask_sum, mask_this;
        int transformed_x, transformed_y;
        for (int x = 0; x < conf.model_side; x++) {
            for (int y = 0; y < conf.model_side; y++) {
                pixel_sum = 0.0;
                mask_sum = 0;
                /* Step through all sub-pixels and add up values */
                for (int xb = 0; xb < conf.image_binning; xb++) {
                    for (int yb = 0; yb < conf.image_binning; yb++) {
                        transformed_x = sp_image_x(img)/2 - (conf.model_side/2)*conf.image_binning + x*conf.image_binning + xb;
                        transformed_y = sp_image_y(img)/2 - (conf.model_side/2)*conf.image_binning + y*conf.image_binning + yb;
                        if (transformed_x >= 0 && transformed_x < sp_image_x(img) &&
                                transformed_y >= 0 && transformed_y < sp_image_y(img)) {
                            pixel_this = sp_cabs(sp_image_get(img, transformed_x, transformed_y, 0)) ;
                            mask_this = sp_image_mask_get(img,  transformed_x, transformed_y, 0);
                        } else {
                            pixel_this = 0.;
                            mask_this = 0;
                        }
                        if (mask_this > 0) {
                            pixel_sum += pixel_this;
                            mask_sum += 1;
                        }
                    }
                }
                /* As long as there were at least one subpixel contributin to the
                 pixel (that is that was not masked out) we include the data and
                 don't mask out the pixel. */
                if (mask_sum > 0) {
                 //for tmp use
                    sp_matrix_set(images[i],x,y,pixel_sum/(real)mask_sum);
                    sp_imatrix_set(masks[i],x,y,1);
                } else {
                    sp_matrix_set(images[i],x,y,0.);
                    sp_imatrix_set(masks[i],x,y,0);
                }
            }
        }
        sp_image_free(img);
    }
    return images;
}

/* Read the common mask in a similar way as in which the individual
 masks are read. The image values of the spimage file are used as
 the masks and the mask of the image is never read. Values of 0 are
 masked out. */
sp_imatrix *read_mask(Configuration conf)
{
    sp_imatrix *mask = sp_imatrix_alloc(conf.model_side,conf.model_side);;
    Image *mask_in = sp_image_read(conf.mask_file,0);
    /* read and rescale mask */
    int mask_sum;
    int this_value;
    int transformed_x, transformed_y;
    for (int x = 0; x < conf.model_side; x++) {
        for (int y = 0; y < conf.model_side; y++) {
            mask_sum = 0;
            for (int xb = 0; xb < conf.image_binning; xb++) {
                for (int yb = 0; yb < conf.image_binning; yb++) {
                    transformed_x = sp_image_x(mask_in)/2 - (conf.model_side/2)*conf.image_binning + x*conf.image_binning + xb;
                    transformed_y = sp_image_y(mask_in)/2 - (conf.model_side/2)*conf.image_binning + y*conf.image_binning + yb;
                    if (transformed_x >= 0 && transformed_x < sp_image_x(mask_in) &&
                            transformed_y >= 0 && transformed_y < sp_image_y(mask_in)) {
                        this_value = sp_cabs(sp_image_get(mask_in, transformed_x, transformed_y, 0));
                    } else {
                        this_value = 0;
                    }
                    if (this_value) {
                        mask_sum += 1;
                    }
                }
            }
            if (mask_sum > 0) {
                sp_imatrix_set(mask,x,y,1);
            } else {
                sp_imatrix_set(mask,x,y,0);
            }
        }
    }
    sp_image_free(mask_in);
    
    /* Also mask out everything outside a sphere with diameter
     conf.model_side. In this way we avoid any bias from the
     alignment of the edges of the model. */
    for (int x = 0; x < conf.model_side; x++) {
        for (int y = 0; y < conf.model_side; y++) {
            if (sqrt(pow((real)x - (real)conf.model_side/2.0+0.5,2) +
                     pow((real)y - (real)conf.model_side/2.0+0.5,2)) >
                    conf.model_side/2.0) {
                sp_imatrix_set(mask,x,y,0);
            }
        }
    }
    return mask;
}


/* This function writes some run information to a file and closes the
 file. This is intended to contain things that don't change during the
 run, currently number of images, the random seed and wether compact_output
 is used. This file is used by the viewer. */
void write_run_info(char *filename, Configuration conf, int random_seed) {
    hid_t file_id, space_id, dataset_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    space_id = H5Screate(H5S_SCALAR);
    dataset_id = H5Dcreate1(file_id, "/number_of_images", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.number_of_images);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    
    space_id = H5Screate(H5S_SCALAR);
    dataset_id = H5Dcreate1(file_id, "/compact_output", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.compact_output);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    
    space_id = H5Screate(H5S_SCALAR);
    dataset_id = H5Dcreate1(file_id, "/random_seed", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &random_seed);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    
    space_id = H5Screate(H5S_SCALAR);
    dataset_id = H5Dcreate1(file_id, "/recover_scaling", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conf.recover_scaling);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    
    H5Fclose(file_id);
}

/* The state file contains info about the run but opposed to the run info
 this contains things that change during the run. Currently it only
 contains the current iteration. This file is used by the viewer and
 is important to be able to view the data while the program is still
 running. */
hid_t open_state_file(char *filename) {
    hid_t file_id, space_id, dataset_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    space_id = H5Screate(H5S_SCALAR);
    dataset_id = H5Dcreate1(file_id, "/iteration", H5T_NATIVE_INT, space_id, H5P_DEFAULT);
    int iteration_start_value = -1;
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration_start_value);
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    
    H5Fflush(file_id, H5F_SCOPE_GLOBAL);
    
    return file_id;
}

/* Write new information to the state file. Call this at the beginning of each iteration. */
void write_state_file_iteration(hid_t file_id, int iteration) {
    hid_t dataset_id;
    hsize_t file_size;
    H5Fget_filesize(file_id, &file_size);
    
    dataset_id = H5Dopen(file_id, "/iteration", H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &iteration);
    H5Fflush(dataset_id, H5F_SCOPE_LOCAL);
    H5Dclose(dataset_id);
    H5Fflush(file_id, H5F_SCOPE_GLOBAL);
}

/* Close state file (duh). */
void close_state_file(hid_t file_id) {
    H5Fclose(file_id);
}

/* The rotation sampling is tabulated to save computational time (although i think
 the new code is fast enough so this is a bit ridicculus). This file reads the
 rotations from file and returns them and the weights in the respective input
 pointers and returns the number of rotations. */
int read_rotations_file(const char *filename, Quaternion **rotations, real **weights) {
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset_id = H5Dopen1(file_id, "/rotations");
    hid_t space_id = H5Dget_space(dataset_id);
    hsize_t dims[2];
    hsize_t maxdims[2];
    H5Sget_simple_extent_dims(space_id, dims, maxdims);
    
    int N_slices = dims[0];
    
    real *input_array = (real*) malloc(N_slices*5*sizeof(real));
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, input_array);
    
    *rotations = (Quaternion*)malloc(N_slices*sizeof(Quaternion));
    *weights = (real*)malloc(N_slices*sizeof(real));
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
        /*
         Quaternion *this_quaternion = quaternion_alloc();
         memcpy(this_quaternion->q, &input_array[i_slice*5], 4*sizeof(real));
         (*rotations)[i_slice] = this_quaternion;
         (*weights)[i_slice] = input_array[i_slice*5+4];
         */
        Quaternion this_quaternion;
        memcpy(this_quaternion.q, &input_array[i_slice*5], 4*sizeof(real));
        (*rotations)[i_slice] = this_quaternion;
        (*weights)[i_slice] = input_array[i_slice*5+4];
    }
    return N_slices;
}

/* Create a directory or do nothing if the directory already exists. This
 function also works if there are several levels of diretcories that does
 not exist. */
void mkdir_recursive(const char *dir, int permission) {
    char tmp[PATH_MAX];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", dir);
    len = strlen(tmp);
    if (tmp[len-1] == '/') {
        tmp[len-1] = 0;
    }
    for (p = tmp+1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, permission);
            *p = '/';
        }
    }
    mkdir(tmp, permission);
}


int load_selected_images_by_log (Configuration conf, FILE * logF,  sp_matrix **images, sp_imatrix ** masks,
                                 imageInd list, const char* log_format, const char* file_path_format, real error_lim, long int current_time,
                                 int imax){
    // log format FILEindex, time_stamp, categrade, classification error, others.

    long int classification_time = 0;
    int categrade = -1 ;
    real error = -1;
    long int file_index = -1 ;
    int rot = -1;
    int rerr =   fscanf (logF, log_format, &file_index,&classification_time,&rot,&categrade, &error );
    printf("%d\n\n",rerr);
    sp_matrix* image = (sp_matrix*) sp_matrix_alloc(conf.model_side,conf.model_side);
    sp_imatrix* mask = (sp_imatrix*) sp_imatrix_alloc(conf.model_side,conf.model_side);
    int N_load = 0;
    imageInd::iterator plist = list.begin();
    
    while (rerr != EOF || rerr > 8){
            /*printf("Debug load_selected_images_by_log log_format=%s file_path_format=%s error_lim=%g current_time=%ld imax=%d classification_time=%ld categrade=%d error=%g file_index=%ld rerr=%d", log_format,file_path_format,error_lim, current_time, imax, classification_time, categrade,error,
               file_index, rerr);*/
        
        if (N_load > imax)
            break;
    
        if (classification_time < current_time ){
            if ( error < error_lim && categrade == 1){
                // read images
                read_single_image( conf,  image, mask,file_path_format, file_index);
                images[*plist] = image;
                masks[*plist] = mask;
                plist++;
                N_load ++;
            }
            rerr =   fscanf (logF, log_format, &file_index,&classification_time,&rot,&categrade, &error );
        }
        else break;
        }
        return N_load;
}


int read_single_image(Configuration conf, sp_matrix * im, sp_imatrix* msk, const char* file_path_format, long int file_index){
    Image *img;
    char filename_buffer[PATH_MAX];
    printf("Debug...in read_single_image\n\n");
    printf("%s\n\n\n\n",filename_buffer);

    sprintf(filename_buffer,"%s%.4d.h5", conf.image_prefix,file_index);
    img = sp_image_read(filename_buffer,0);
    if (img ==NULL)
        return -1;

    /* The algorithm can't handle negative data so if we have negative
         values they are simply set to 0. */
    for (int pixel_i = 0; pixel_i < sp_image_size(img); pixel_i++) {
        if (sp_real(img->image->data[pixel_i]) < 0.) {
            sp_real(img->image->data[pixel_i]) = 0.;
        }
    }

    /* Downsample images and masks by nested loops of all
         downsampled pixels and then all subpixels. */
    real pixel_sum, pixel_this;
    int mask_sum, mask_this;
    int transformed_x, transformed_y;
    for (int x = 0; x < conf.model_side; x++) {
        for (int y = 0; y < conf.model_side; y++) {
            pixel_sum = 0.0;
            mask_sum = 0;
            /* Step through all sub-pixels and add up values */
            for (int xb = 0; xb < conf.image_binning; xb++) {
                for (int yb = 0; yb < conf.image_binning; yb++) {
                    transformed_x = sp_image_x(img)/2 - (conf.model_side/2)*conf.image_binning + x*conf.image_binning + xb;
                    transformed_y = sp_image_y(img)/2 - (conf.model_side/2)*conf.image_binning + y*conf.image_binning + yb;
                    if (transformed_x >= 0 && transformed_x < sp_image_x(img) &&
                            transformed_y >= 0 && transformed_y < sp_image_y(img)) {
                        pixel_this = sp_cabs(sp_image_get(img, transformed_x, transformed_y, 0));
                        mask_this = sp_image_mask_get(img,  transformed_x, transformed_y, 0);
                    } else {
                        pixel_this = 0.;
                        mask_this = 0;
                    }
                    if (mask_this > 0) {
                        pixel_sum += pixel_this;
                        mask_sum += 1;
                    }
                }
            }
            /* As long as there were at least one subpixel contributin to the
                 pixel (that is that was not masked out) we include the data and
                 don't mask out the pixel. */
            if (mask_sum > 0) {
                sp_matrix_set(im,x,y,pixel_sum/(real)mask_sum);
                sp_imatrix_set(msk,x,y,1);
            } else {
                sp_matrix_set(im,x,y,0.);
                sp_imatrix_set(msk,x,y,0);
            }
        }
    }
    sp_image_free(img);
    return 0;
}


