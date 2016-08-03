cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
"${CMAKE_SOURCE_DIR}/cmake/Modules/")

IF(CMAKE_COMPILER_IS_GNUCC)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  # Necessary due to CUDA compatibility problems
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  # The CUDA module thinks cc is gcc and passes dumpspecs.
  # This is a workaround
  if(CUDA_HOST_COMPILER STREQUAL "/usr/bin/cc")
    SET(CUDA_HOST_COMPILER "/usr/bin/clang")
  endif()
endif()

FIND_PACKAGE(SpImage REQUIRED)
FIND_PACKAGE(HDF5 REQUIRED)
	
FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-arch sm_20")

find_package(GSL REQUIRED)

list(APPEND INCLUDE_LIST ${SPIMAGE_INCLUDE_DIR} ${HDF5_INCLUDE_DIR} ${LIBCONFIG_INCLUDE_DIR})

set(LOCAL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LOCAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

#message(FATAL_ERROR ${LIBCONFIG_LIBRARIES})
#message(FATAL_ERROR ${SPIMAGE_LIBRARIES})

add_subdirectory(src)
