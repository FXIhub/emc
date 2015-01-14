cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

#message("${LOCAL_SRC}/rotations.c")

cuda_add_executable(generate_rotations generate_rotations.c ${LOCAL_SRC}/rotations.c)
target_link_libraries(generate_rotations spimage config gsl gslcblas hdf5)

cuda_add_executable(modify_configuration modify_configuration.c)
target_link_libraries(modify_configuration config)

# add the install target
install(TARGETS generate_rotations DESTINATION bin)
install(TARGETS modify_configuration DESTINATION bin)
