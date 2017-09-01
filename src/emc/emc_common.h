#ifndef EMC_COMMON_H
#define EMC_COMMON_H


#include <spimage.h>
#include <rotations.h>
#include <configuration.h>
#include <stdio.h>
#include <stdlib.h>

//typedef real float;

typedef struct{
  int side;
  real wavelength;
  real pixel_size;
  real detector_distance;
}Setup;

//#define PATH_MAX = 256;

#endif
