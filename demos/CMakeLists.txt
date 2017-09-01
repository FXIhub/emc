cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

include_directories(${LOCAL_INCLUDE})

set(CUDA_NVCC_FLAGS "-arch sm_20” --relocatable-device-code=true ")
cuda_add_executable(emc emc.cpp)

cuda_add_cufft_to_target(emc)

target_link_libraries(emc "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)


cuda_add_executable(emc_dis emc_dis.cpp)

cuda_add_cufft_to_target(emc_dis)

target_link_libraries(emc_dis "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)




cuda_add_executable(emc_pgs emc_pg_sel.cpp)

cuda_add_cufft_to_target(emc_pgs)

target_link_libraries(emc_pgs "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

#cuda_add_executable(ty  try.cpp)

#cuda_add_cufft_to_target(ty)

#target_link_libraries(ty "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

