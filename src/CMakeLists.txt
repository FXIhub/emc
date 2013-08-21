cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

include_directories(${LOCAL_INCLUDE})

add_subdirectory(emc)
add_subdirectory(utils)

# cuda_add_executable(emc emc.c rotations.c configuration.c emc_cuda.cu emc_atomic.cu)
# cuda_add_cufft_to_target(emc)
# target_link_libraries(emc spimage config gsl gslcblas hdf5)

# cuda_add_executable(generate_rotations generate_rotations.c rotations.c)
# target_link_libraries(generate_rotations spimage config gsl gslcblas hdf5)

# cuda_add_executable(modify_configuration modify_configuration.c)
# target_link_libraries(modify_configuration config)
