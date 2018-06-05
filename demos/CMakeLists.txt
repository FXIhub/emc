cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

include_directories(${LOCAL_INCLUDE})

set(CMAKE_CXX_FLAGS "-DOMPI_SKIP_MPICXX")
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30 -rdc=true)
set( CUDA_SEPARABLE_COMPILATION ON)


#cuda_compile(try_o try_performance.cu)
#cuda_add_executable(optEMC opt_emc.cu)
#cuda_add_cufft_to_target(optEMC)
#target_link_libraries(optEMC ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})

#cuda_add_executable(stoEMC StochasticEMC.cu)
#cuda_add_cufft_to_target(stoEMC)
#target_link_libraries(stoEMC ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})


cuda_add_executable(emc emc.cpp)
cuda_add_cufft_to_target(emc)
target_link_libraries(emc "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)


cuda_add_executable(emc_dis emc_dis.cpp)
cuda_add_cufft_to_target(emc_dis)
target_link_libraries(emc_dis "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

#cuda_add_executable(emc_double_mask emc_double_mask.cpp)
#cuda_add_cufft_to_target(emc_double_mask)
#target_link_libraries(emc_double_mask "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

#cuda_add_executable(emc_boot boot_st.cpp)
#cuda_add_cufft_to_target(emc_boot)
#target_link_libraries(emc_boot "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

cuda_add_executable(emc_Cen emc_withCenter.cpp)
cuda_add_cufft_to_target(emc_Cen)
target_link_libraries(emc_Cen "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

#cuda_add_executable(emc_pgs emc_pg_sel.cpp)
#cuda_add_cufft_to_target(emc_pgs)
#target_link_libraries(emc_pgs "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)


#QusiOnline files
#cuda_add_executable(Qusi1 QusiOnline_EMC_V1.cpp)
#cuda_add_cufft_to_target(Qusi1)
#target_link_libraries(Qusi1 ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})


#write test log file
#cuda_add_executable(wlog write_test_log_file.cpp)
#cuda_add_cufft_to_target(wlog)
#target_link_libraries(wlog ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})

#insert frames with given rotations
#cuda_add_executable(insert InsertFramesGivenRotations.cpp)
#cuda_add_cufft_to_target(insert)
#target_link_libraries(insert ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})

cuda_add_executable(insertW InsertFramesWithGivenRotationsAndWeights.cpp)
cuda_add_cufft_to_target(insertW)
target_link_libraries(insertW ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} emcLib ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES})




#cuda_add_executable(ty  try.cpp)

#cuda_add_cufft_to_target(ty)

#target_link_libraries(ty "-dlink"  ${SPIMAGE_LIBRARIES} ${HDF5_LIBRARIES} emcLib)

