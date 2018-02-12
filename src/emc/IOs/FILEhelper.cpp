
#include <FILEhelper.h>

FILE* create_file_descriptor(Configuration conf, const char * name, const char* mode){
    char filename_buffer[256];
    sprintf(filename_buffer, "%s/%s", conf.output_dir,name);
    FILE* file = fopen(filename_buffer, mode);
    if(file !=NULL)
        return file;
    else
        printf("ERROR: cannot open file %s\n", filename_buffer);
}

void write_best_quat(Configuration conf, int iteration, Quaternion* rotations, int* best_rotation, int N_images){
    char filename_buffer[256];
    sprintf(filename_buffer, "%s/best_quaternion_%.4d.data", conf.output_dir,iteration);
    FILE* best_quat_file = fopen(filename_buffer, "wp");
    for (int i_image = 0; i_image < N_images; i_image++) {
        write_quat(best_quat_file, rotations, best_rotation[i_image]);
        //fprintf(best_quat_file, "%g %g %g %g\n", rotations[best_rotation[i_image]].q[0], rotations[best_rotation[i_image]].q[1],
        // rotations[best_rotation[i_image]].q[2], rotations[best_rotation[i_image]].q[3]);
    }
    fclose(best_quat_file);
}

void write_quat(FILE* fl, Quaternion* rotations, int best_rotation){
    fprintf(fl, "%g %g %g %g\n", rotations[best_rotation].q[0], rotations[best_rotation].q[1],
            rotations[best_rotation].q[2], rotations[best_rotation].q[3]);
}

void write_int_array(FILE* file, int* vec, int N){
    for(int i =0; i<N; i++)
        fprintf(file, "%d ", vec[i]);
    fprintf(file, "\n");
    fflush(file);
}
void write_real_array(FILE* file, real* vec, int N){
    for(int i =0; i<N; i++)
        fprintf(file, "%g ", vec[i]);
    fprintf(file, "\n");
    fflush(file);
}

void write_ave_respons(Configuration conf, real* full_respons, int N_images, int N_slices, int iteration,real* average_resp){
    char filename_buffer[256];
    memset ( (void *) (average_resp), 0, sizeof(real) * N_slices);
    for (int i_slice = 0; i_slice < N_slices; i_slice++) {
#pragma omp parallel for
        for (int i_image = 0; i_image < N_images; i_image++) {
            average_resp[i_slice] += full_respons[i_slice*N_images+i_image];
        }
    }
    sprintf(filename_buffer, "%s/average_resp_%.4d.h5", conf.output_dir, iteration);
    write_1d_real_array_hdf5(filename_buffer, average_resp, N_slices);
}

void write_model(Configuration conf, int iteration, int N_model, sp_3matrix *model, sp_3matrix* weight){
    char filename_buffer[256];
    Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
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
}

void write_weight(Configuration conf, int iteration, int N_model, sp_3matrix *weight){
    char filename_buffer[256];
    Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);

    for (int i = 0; i < N_model; i++) {
        model_out->image->data[i] = sp_cinit(weight->data[i], 0.);
        model_out->mask->data[i] = 1;
    }
    sprintf(filename_buffer, "%s/weight_%.4d.h5", conf.output_dir, iteration);
    sp_image_write(model_out, filename_buffer, 0);
}

void write_time(FILE* file, double exeTime, int iteration){
    fprintf(file, "%d %f\n", iteration, exeTime);
    fflush(file);
}

void write_time_by_step(FILE* file, double exeTime, int iteration,const char* step){
    fprintf(file, "%d %f %s\n", iteration, exeTime,step);
    fflush(file);
    fflush(file);
}

void write_final_model(Configuration conf, int N_model, sp_3matrix *model,sp_3matrix *weight){
    char filename_buffer[256];
    Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
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
    sprintf(filename_buffer,"%s/model_final.h5", conf.output_dir);
    sp_image_write(model_out,filename_buffer,0);
}



