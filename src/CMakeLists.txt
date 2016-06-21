cmake_minimum_required(VERSION 2.6)

PROJECT (emc)

include_directories(${INCLUDE_LIST} ${LOCAL_INCLUDE})

add_subdirectory(emc)
add_subdirectory(utils)
