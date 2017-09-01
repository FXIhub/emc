/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */

#ifndef EMC_SYSCON_H
#define EMC_SYSCON_H
/*
#ifdef __cplusplus
extern "C"{
#endif
*/
void cuda_choose_best_device();
int cuda_get_device();
void cuda_set_device(int i_device);
int cuda_get_number_of_devices();
int cuda_get_best_device();
void cuda_print_device_info() ;
/*
#ifdef __cplusplus
}
#endif
*/
#endif
