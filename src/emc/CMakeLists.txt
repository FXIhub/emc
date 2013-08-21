cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

#message("${LOCAL_SRC}/rotations.c")

cuda_add_executable(emc emc.c ${LOCAL_SRC}/rotations.c configuration.c emc_cuda.cu emc_atomic.cu)
cuda_add_cufft_to_target(emc)
target_link_libraries(emc spimage config gsl gslcblas hdf5)
