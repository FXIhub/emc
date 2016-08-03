cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

#message("${LOCAL_SRC}/rotations.c")

cuda_add_executable(emc_generate_rotations generate_rotations.c ${LOCAL_SRC}/rotations.c)
target_link_libraries(emc_generate_rotations ${SPIMAGE_LIBRARIES} ${LIBCONFIG_LIBRARIES} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${HDF5_LIBRARIES})

cuda_add_executable(emc_modify_configuration modify_configuration.c)
target_link_libraries(emc_modify_configuration ${LIBCONFIG_LIBRARIES})

# add the install target
install(TARGETS emc_generate_rotations DESTINATION bin)
install(TARGETS emc_modify_configuration DESTINATION bin)
