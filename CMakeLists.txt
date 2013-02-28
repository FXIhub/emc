cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
"${CMAKE_SOURCE_DIR}/cmake/Modules/")

#message(${CMAKE_MODULE_PATH})

IF(CMAKE_COMPILER_IS_GNUCC)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

FIND_PACKAGE(SpImage REQUIRED)
FIND_PACKAGE(HDF5 REQUIRED)
	
FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-arch sm_20")
#cuda_add_library(_emc emc.c rotations.c emc_cuda.cu emc_atomic.cu)
#cuda_add_cufft_to_target(_emc)

find_package(GSL REQUIRED)

#add_library(rotations rotations.c)
#cuda_add_library(emc_cuda emc_cuda.cu emc_atomic.cu)
#add_executable(emc emc.c rotations.c)
#target_link_libraries(emc emc_cuda spimage config rotations gsl gslcblas)


cuda_add_executable(emc emc.c rotations.c emc_cuda.cu emc_atomic.cu)
cuda_add_cufft_to_target(emc)
target_link_libraries(emc spimage config gsl gslcblas hdf5)

