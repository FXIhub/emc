cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

#message("${LOCAL_SRC}/rotations.c")

#include_directories(${INCLUDE_LIST} ${LOCAL_INCLUDE})

cuda_add_executable(emc emc.c ${LOCAL_SRC}/rotations.c configuration.c emc_cuda.cu emc_atomic.cu)
cuda_add_cufft_to_target(emc)
target_link_libraries(emc ${SPIMAGE_LIBRARIES} config ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${HDF5_LIBRARIES})

# add the install target
install(TARGETS emc DESTINATION bin)
install(FILES "${LOCAL_SRC}/emc/emc.h"
         DESTINATION include)
install(FILES "${LOCAL_SRC}/emc/configuration.h"
         DESTINATION include)
