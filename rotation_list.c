#include <spimage.h>
#include "rotations.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Error: must provide a n number and a filename.\n");
    exit(1);
  }
  int n = atoi(argv[1]);
  char * filename = argv[2];
  Quaternion **q_list;
  real *weights;
  const long long int N_slices = generate_rotation_list(n, &q_list, &weights);

  FILE *f = fopen(filename,"wp");
  for (long long int i = 0; i < N_slices; i++) {
    fprintf(f, "%g %g %g %g\n", q_list[i]->q[0], q_list[i]->q[1],
	    q_list[i]->q[2], q_list[i]->q[3]);
  }
  fclose(f);
}
