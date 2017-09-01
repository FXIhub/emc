#ifndef MPIHELPER_H
#define MPIHELPER
#include <mpi.h>
#include <configuration.h>
#include <spimage.h>
#define MPI_EMC_PRECISION MPI_FLOAT


void Global_Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype, MPI_Op, MPI_Comm);
void Global_Gatherv(const void *, int , MPI_Datatype , void *, const int *, const int *,int, MPI_Comm);

void Broadcast_Model(sp_3matrix *model, int taskid);
void Broadcast_3matrix(sp_3matrix *ma, int taskid);


#endif // MPIHELPER_H
