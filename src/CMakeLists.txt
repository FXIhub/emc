cmake_minimum_required(VERSION 2.6)

PROJECT (emc)


include_directories(${LOCAL_INCLUDE})

file(GLOB emc_C_source  emc/Common/*.cpp emc/Configurations/*.cpp emc/IOs/*.cpp emc/MPIs/*.cpp ) 
file(GLOB emc_source  emc/CUDA/*.cu)
set(CUDA_NVCC_FLAGS "-arch=sm_20" "--relocatable-device-code=true")
SET(CUDA_SEPARABLE_COMPILATION ON)


cuda_add_library(rotations utils/rotations.cpp utils/rotation_list.c)
target_link_libraries(rotations ${SPIMAGE_LIBRARIES} config ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${HDF5_LIBRARIES})

cuda_add_library(emcLib ${emc_source} ${emc_C_source})
target_link_libraries(emcLib  config rotations ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}  ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${MPI_LIBRARIES})
