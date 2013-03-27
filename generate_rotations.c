#include <hdf5.h>
#include <getopt.h>
#include <ctype.h>
#include "rotations.h"

int main(int argc, char **argv) {
  int c;
  int n;
  char *filename;
  filename = NULL;
  while ((c = getopt(argc, argv, "f:")) != -1) {
    switch(c) {
    case 'f':
      filename = optarg;
      break;
    case 'h':
      printf("Usage generate_rotations [-f FILENAME] N");
      exit(0);
    case '?':
      if (optopt == 'f')
	fprintf(stderr, "Option -f requires an argument\n");
      else if (isprint(optopt))
	fprintf(stderr, "Unknown option -%c\n", optopt);
      else
	fprintf(stderr, "Unknown option character\n");
    default:
      abort();
    }
  }
  n = atoi(argv[optind]);
  if (filename == NULL) {
    filename = malloc(100);
    sprintf(filename, "rotations_%d.h5", n);
  }

  Quaternion **rotations;
  real *weights;
  const int N_slices = generate_rotation_list(n,&rotations,&weights);
  
  //The output array will have N_slices rows of 5 elements: the quaternion followed by the weight.
  real *out_buffer = malloc(N_slices*5*sizeof(real));

  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
    memcpy(&out_buffer[i_slice*5], rotations[i_slice]->q, 4*sizeof(real));
    out_buffer[i_slice*5+4] = weights[i_slice];
    quaternion_free(rotations[i_slice]);
  }
  free(rotations);
  free(weights);

  hsize_t dim[2];
  dim[0] = N_slices;
  dim[1] = 5;
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t space_id = H5Screate_simple(2, dim, NULL);
  hid_t dataset_id = H5Dcreate1(file_id, "/rotations", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT);
  H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_buffer);
  free(out_buffer);
  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Fclose(file_id);
}
