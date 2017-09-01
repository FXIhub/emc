#ifndef FILE_HELPER_H
#define FILE_HELPER_H
#include <iostream>
#include <fstream>
#include <spimage.h>
#include <rotations.h>
#include <configuration.h>
#include <IOHDF5.h>
#include <string.h>

void write_3matrix(sp_3matrix * model, Configuration, char*);


FILE* create_file_descriptor(Configuration conf, const char * name, const char* mode);

void write_best_quat(Configuration conf, int iteration, Quaternion* rotations, int* best_rotation, int N_images);

void write_int_array(FILE* file, int* vec, int N);
void write_real_array(FILE* file, real* vec, int N);
void write_ave_respons(Configuration conf, real* full_respons, int N_images, int N_slices, int iteration ,real* average_resp);
void write_model(Configuration conf, int iteration, int N_model, sp_3matrix *model, sp_3matrix* weight);
void write_final_model(Configuration conf, int N_model, sp_3matrix *model,sp_3matrix *weight);
void write_weight(Configuration conf, int iteration, int N_model, sp_3matrix *weight);

void write_time(FILE* file, double exeTime, int iteration);
void write_quat(FILE* fl, Quaternion* rotations, int best_rotation);

#endif // FILE_HELPER_H