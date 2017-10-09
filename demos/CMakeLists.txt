cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

include_directories(${LOCAL_INCLUDE})

set(CMAKE_CXX_FLAGS "-DOMPI_SKIP_MPICXX")
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30 -rdc=true)
set( CUDA_SEPARABLE_COMPILATION ON)


cuda_add_executable(emc emc.cpp)
cuda_add_cufft_to_target(emc)
target_link_libraries(emc "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)


cuda_add_executable(emc_dis emc_dis.cpp)
cuda_add_cufft_to_target(emc_dis)
target_link_libraries(emc_dis "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)




cuda_add_executable(emc_pgs emc_pg_sel.cpp)
cuda_add_cufft_to_target(emc_pgs)
target_link_libraries(emc_pgs "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)



#cuda_compile(try_o try_performance.cu)
cuda_add_executable(tryP try_performance.cu)
cuda_add_cufft_to_target(tryP)
target_link_libraries(tryP ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})


#cuda_add_executable(ty  try.cpp)

#cuda_add_cufft_to_target(ty)

#target_link_libraries(ty "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

