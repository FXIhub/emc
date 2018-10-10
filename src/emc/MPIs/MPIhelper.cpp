/*Author :Jing Liu
 Modified  2013-8-12 from C++ version to c version
 */
#include "MPIhelper.h"
#include <string.h>

void Global_Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype ty,
                      MPI_Op op , MPI_Comm comm){    
    int error = MPI_Allreduce((void*)sendbuf,(void*)recvbuf,count,ty,op,comm);
    if (error != MPI_SUCCESS)
        printf("MPI error (Global_Allreduce): %d\n",error);
}

void Global_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, const int *recvcounts, const int *displs,
                    int root, MPI_Comm comm){
    int error = MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf,
                            recvcounts, displs, sendtype, root, comm);
    if (error != MPI_SUCCESS)
        printf("MPI error (Global_Gatherv): %d\n",error);
}


void Broadcast_Model(sp_3matrix *model, int taskid){
    Broadcast_3matrix(model, taskid);
}
void Broadcast_3matrix(sp_3matrix *ma, int root){
    real * tmp = ma->data;
    ma->data = NULL;
    MPI_Bcast(&(ma->x),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    MPI_Bcast(&(ma->y),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    MPI_Bcast(&(ma->z),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
        // COMM_WORLD.Bcast(ma, 1,MPI_SP_3MATRIX, taskid);
    MPI_Bcast(tmp, (ma->x)*(ma->y)*(ma->z),MPI_INT, root,MPI_COMM_WORLD);
    ma->data = tmp;
}

void MPI_Broadcast_int_list(int* lst, int N_images,int root){
    MPI_Bcast(lst,N_images,MPI_INT,root,MPI_COMM_WORLD);
    printf("broadcasting list done!\n");
}
