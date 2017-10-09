#README#

This is a software for 3D reconstruction of single particle XFEL imaging using multiple GPUs / GPUs cluster. 


1) Software dependency.
Except from the standard CUDA and C++ libary, this software also relies on HDF5, GSL and SpImage (git://github.com/FilipeMaia/libspimage).

2) Installation.
CMake and ccmake is required for installation.

3) Run multiple GPU version
This code use emc.conf as default configuration file. After compile the multiple-GPUs EMC code will generate an executable file emc_dis, to run it, you have to have emc.conf in the same directory of emc_dis. 
To run emc_dis with MPI, use the following command: mpirun -np 4 ./emc_dis


