/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#include<emc_cuda_ec.h>

__device__ void cuda_get_slice(real *model, real *slice,
                               real *rot, real *x_coordinates,
                               real *y_coordinates, real *z_coordinates, int slice_rows,
                               int slice_cols, int model_x, int model_y, int model_z,
                               int tid, int step)
{
    const int x_max = slice_rows;
    const int y_max = slice_cols;
    //tabulate angle later
    real new_x, new_y, new_z;
    int round_x, round_y, round_z;
    real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
    real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
    real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
    real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
    real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
    real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
    real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
    real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
    real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
    for (int x = 0; x < x_max; x++) {
        for (int y = tid; y < y_max; y+=step) {
            /* This is just a matrix multiplication with rot */
            new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x];
            new_y = m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x];
            new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x];
            /* changed the next lines +0.5 -> -0.5 (11 dec 2012)*/
            round_x = lroundf(model_x/2.0f - 0.5f + new_x);
            round_y = lroundf(model_y/2.0f - 0.5f + new_y);
            round_z = lroundf(model_z/2.0f - 0.5f + new_z);
            if (round_x > 0 && round_x < model_x &&
                    round_y > 0 && round_y < model_y &&
                    round_z > 0 && round_z < model_z) {
                slice[y*x_max+x] = model[round_z*model_x*model_y + round_y*model_x + round_x];
            }else{
                slice[y*x_max+x] = -1.0f;
            }
        }
    }
}

__device__ real interpolate_model_get(real *model, int model_x, int model_y, int model_z, real new_x, real new_y, real new_z) {
    real interp_sum, interp_weight;
    real weight_x, weight_y, weight_z;
    int index_x, index_y, index_z;
    real low_weight_x, low_weight_y, low_weight_z;
    int low_x, low_y, low_z;
    int out_of_range = 0;

    if (new_x > -0.5 && new_x <= 0.) {
        low_weight_x = 0.;
        low_x = -1;
    } else if (new_x > 0. && new_x <= (model_x-1)) {
        low_weight_x = ceil(new_x) - new_x;
        low_x = (int)ceil(new_x) - 1;
    } else if (new_x > (model_x-1) && new_x < (model_x-0.5)) {
        low_weight_x = 1.;
        low_x = model_x-1;
    } else {
        out_of_range = 1;
    }

    if (new_y > -0.5 && new_y <= 0.) {
        low_weight_y = 0.;
        low_y = -1;
    } else if (new_y > 0. && new_y <= (model_y-1)) {
        low_weight_y = ceil(new_y) - new_y;
        low_y = (int)ceil(new_y) - 1;
    } else if (new_y > (model_y-1) && new_y < (model_y-0.5)) {
        low_weight_y = 1.;
        low_y = model_y-1;
    } else {
        out_of_range = 1;
    }

    if (new_z > -0.5 && new_z <= 0.) {
        low_weight_z = 0.;
        low_z = -1;
    } else if (new_z > 0. && new_z <= (model_z-1)) {
        low_weight_z = ceil(new_z) - new_z;
        low_z = (int)ceil(new_z) - 1;
    } else if (new_z > (model_z-1) && new_z < (model_z-0.5)) {
        low_weight_z = 1.;
        low_z = model_z-1;
    } else {
        out_of_range = 1;
    }

    if (out_of_range == 0) {

        interp_sum = 0.;
        interp_weight = 0.;
        for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
            if (index_x == low_x && low_weight_x == 0.) continue;
            if (index_x == (low_x+1) && low_weight_x == 1.) continue;
            if (index_x == low_x) weight_x = low_weight_x;
            else weight_x = 1. - low_weight_x;

            for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
                if (index_y == low_y && low_weight_y == 0.) continue;
                if (index_y == (low_y+1) && low_weight_y == 1.) continue;
                if (index_y == low_y) weight_y = low_weight_y;
                else weight_y = 1. - low_weight_y;

                for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
                    if (index_z == low_z && low_weight_z == 0.) continue;
                    if (index_z == (low_z+1) && low_weight_z == 1.) continue;
                    if (index_z == low_z) weight_z = low_weight_z;
                    else weight_z = 1. - low_weight_z;

                    if (model[model_x*model_y*index_z + model_x*index_y + index_x] >= 0.) {
                        interp_sum += weight_x*weight_y*weight_z*model[model_x*model_y*index_z + model_x*index_y + index_x];
                        interp_weight += weight_x*weight_y*weight_z;
                    }
                }
            }
        }
        if (interp_weight > 0.) {
            return interp_sum / interp_weight;
        } else {
            return -1.0f;
        }
    } else {
        return -1.0f;
    }
}

__device__ void cuda_get_slice_interpolate(real *model, real *slice, real *rot,
                                           real *x_coordinates, real *y_coordinates, real *z_coordinates,
                                           int slice_rows, int slice_cols, int model_x, int model_y, int model_z,
                                           int tid, int step) {
    const int x_max = slice_rows;
    const int y_max = slice_cols;
    //tabulate angle later
    real new_x, new_y, new_z;

    real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
    real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
    real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
    real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
    real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
    real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
    real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
    real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
    real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
    for (int x = 0; x < x_max; x++) {
        for (int y = tid; y < y_max; y+=step) {
            /* This is just a matrix multiplication with rot */
            new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x] + model_x/2.0 - 0.5;
            new_y = m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x] + model_y/2.0 - 0.5;
            new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x] + model_z/2.0 - 0.5;

            slice[y*x_max+x] = interpolate_model_get(model, model_x, model_y, model_z, new_x, new_y, new_z);
        }
    }
}

__device__ void cuda_insert_slice_interpolate(real *model, real *weight, real *slice,
                                              int * mask, real w, real *rot, real *x_coordinates,
                                              real *y_coordinates, real *z_coordinates, int slice_rows,
                                              int slice_cols, int model_x, int model_y, int model_z,
                                              int tid, int step)
{
    const int x_max = slice_rows;
    const int y_max = slice_cols;
    //tabulate angle later
    real new_x, new_y, new_z;
    real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
    real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
    real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
    real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
    real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
    real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
    real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
    real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
    real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
    for (int x = 0; x < x_max; x++) {
        for (int y = tid; y < y_max; y+=step) {
            //if (mask[y*x_max + x] == 1) {
            if (mask[y*x_max + x] == 1 && slice[y*x_max + x] >= 0.0) {
                /* This is just a matrix multiplication with rot */
                new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x] + model_x/2.0 - 0.5;
                new_y =	m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x] + model_y/2.0 - 0.5;
                new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x] + model_z/2.0 - 0.5;

                interpolate_model_set(model, weight, model_x, model_y, model_z, new_x, new_y, new_z, slice[y*x_max + x], w);
            }
        }
    }
}

__device__ void interpolate_model_set(real *model, real *model_weight, int model_x, int model_y, int model_z,
                                      real new_x, real new_y, real new_z, real value, real value_weight)
{
    real weight_x, weight_y, weight_z;
    int index_x, index_y, index_z;
    real low_weight_x, low_weight_y, low_weight_z;
    int low_x, low_y, low_z;
    int out_of_range = 0;

    if (new_x > -0.5 && new_x <= 0.) {
        low_weight_x = 0.;
        low_x = -1;
    } else if (new_x > 0. && new_x <= (model_x-1)) {
        low_weight_x = ceil(new_x) - new_x;
        low_x = (int)ceil(new_x) - 1;
    } else if (new_x > (model_x-1) && new_x < (model_x-0.5)) {
        low_weight_x = 1.;
        low_x = model_x-1;
    } else {
        out_of_range = 1;
    }

    if (new_y > -0.5 && new_y <= 0.) {
        low_weight_y = 0.;
        low_y = -1;
    } else if (new_y > 0. && new_y <= (model_y-1)) {
        low_weight_y = ceil(new_y) - new_y;
        low_y = (int)ceil(new_y) - 1;
    } else if (new_y > (model_y-1) && new_y < (model_y-0.5)) {
        low_weight_y = 1.;
        low_y = model_y-1;
    } else {
        out_of_range = 1;
    }

    if (new_z > -0.5 && new_z <= 0.) {
        low_weight_z = 0.;
        low_z = -1;
    } else if (new_z > 0. && new_z <= (model_z-1)) {
        low_weight_z = ceil(new_z) - new_z;
        low_z = (int)ceil(new_z) - 1;
    } else if (new_z > (model_z-1) && new_z < (model_z-0.5)) {
        low_weight_z = 1.;
        low_z = model_z-1;
    } else {
        out_of_range = 1;
    }

    if (out_of_range == 0) {

        for (index_x = low_x; index_x <= low_x+1; index_x += 1) {
            if (index_x == low_x && low_weight_x == 0.) continue;
            if (index_x == (low_x+1) && low_weight_x == 1.) continue;
            if (index_x == low_x) weight_x = low_weight_x;
            else weight_x = 1. - low_weight_x;

            for (index_y = low_y; index_y <= low_y+1; index_y += 1) {
                if (index_y == low_y && low_weight_y == 0.) continue;
                if (index_y == (low_y+1) && low_weight_y == 1.) continue;
                if (index_y == low_y) weight_y = low_weight_y;
                else weight_y = 1. - low_weight_y;

                for (index_z = low_z; index_z <= low_z+1; index_z += 1) {
                    if (index_z == low_z && low_weight_z == 0.) continue;
                    if (index_z == (low_z+1) && low_weight_z == 1.) continue;
                    if (index_z == low_z) weight_z = low_weight_z;
                    else weight_z = 1. - low_weight_z;

#if __CUDA_ARCH__ >= 200
                    atomicAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value*value_weight);
                    atomicAdd(&model_weight[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value_weight);
#else
                    atomicFloatAdd(&model[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value*value_weight);
                    atomicFloatAdd(&model_weight[model_x*model_y*index_z + model_x*index_y + index_x], weight_x*weight_y*weight_z*value_weight);
#endif
                }
            }
        }
    }
}

__device__ void cuda_insert_slice(real *model, real *weight, real *slice,
                                  int * mask, real w, real *rot, real *x_coordinates,
                                  real *y_coordinates, real *z_coordinates, int slice_rows,
                                  int slice_cols, int model_x, int model_y, int model_z,
                                  int tid, int step)
{
    const int x_max = slice_rows;
    const int y_max = slice_cols;
    //tabulate angle later
    real new_x, new_y, new_z;
    int round_x, round_y, round_z;
    real m00 = rot[0]*rot[0] + rot[1]*rot[1] - rot[2]*rot[2] - rot[3]*rot[3];
    real m01 = 2.0f*rot[1]*rot[2] - 2.0f*rot[0]*rot[3];
    real m02 = 2.0f*rot[1]*rot[3] + 2.0f*rot[0]*rot[2];
    real m10 = 2.0f*rot[1]*rot[2] + 2.0f*rot[0]*rot[3];
    real m11 = rot[0]*rot[0] - rot[1]*rot[1] + rot[2]*rot[2] - rot[3]*rot[3];
    real m12 = 2.0f*rot[2]*rot[3] - 2.0f*rot[0]*rot[1];
    real m20 = 2.0f*rot[1]*rot[3] - 2.0f*rot[0]*rot[2];
    real m21 = 2.0f*rot[2]*rot[3] + 2.0f*rot[0]*rot[1];
    real m22 = rot[0]*rot[0] - rot[1]*rot[1] - rot[2]*rot[2] + rot[3]*rot[3];
    for (int x = 0; x < x_max; x++) {
        for (int y = tid; y < y_max; y+=step) {
            //if (mask[y*x_max + x] == 1) {
            if (mask[y*x_max + x] == 1 && slice[y*x_max + x] >= 0.0) {
                /* This is just a matrix multiplication with rot */
                new_x = m00*z_coordinates[y*x_max+x] + m01*y_coordinates[y*x_max+x] + m02*x_coordinates[y*x_max+x];
                new_y =	m10*z_coordinates[y*x_max+x] + m11*y_coordinates[y*x_max+x] + m12*x_coordinates[y*x_max+x];
                new_z = m20*z_coordinates[y*x_max+x] + m21*y_coordinates[y*x_max+x] + m22*x_coordinates[y*x_max+x];
                /* changed the next lines +0.5 -> -0.5 (11 dec 2012)*/
                round_x = lroundf(model_x/2.0f - 0.5f + new_x);
                round_y = lroundf(model_y/2.0f - 0.5f + new_y);
                round_z = lroundf(model_z/2.0f - 0.5f + new_z);
                if (round_x >= 0 && round_x < model_x &&
                        round_y >= 0 && round_y < model_y &&
                        round_z >= 0 && round_z < model_z) {
                    /* this is a simple compile time check that can go bad at runtime, but such is life */
#if __CUDA_ARCH__ >= 200
                    atomicAdd(&model[round_z*model_x*model_y + round_y*model_x + round_x], w * slice[y*x_max + x]);
                    atomicAdd(&weight[round_z*model_x*model_y + round_y*model_x + round_x], w);
#else
                    atomicFloatAdd(&model[round_z*model_x*model_y + round_y*model_x + round_x], w * slice[y*x_max + x]);
                    atomicFloatAdd(&weight[round_z*model_x*model_y + round_y*model_x + round_x], w);
#endif
                }
            }
        }
    }
}

