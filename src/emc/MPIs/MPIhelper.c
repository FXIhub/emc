/*Author :Jing Liu
*/
#include <MPIhelper.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define MPI_EMC_PRECISION MPI_FLOAT

// to be checked
void MPI_Create_Configuration(MPI_Datatype* MPI_CONFIG){
    printf("Creating Configuration!... ");
    MPI_Datatype types[20] = {MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT
                              ,MPI_INT, MPI_INT, MPI_INT,
                              MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT,
                              MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,
                              MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
    int blocklen[20] = {1,1,1,1,1,1,1,1,1,1,1,1,1
                        ,1,1,1,1,1,1,1};
    MPI_Aint disp[20];
    disp[0] = 0;
    for(int i =1; i<13; i++){
        disp[i] =sizeof(int) + disp[i-1];
    }
    for(int i =13; i<20; i++){
        disp[i] =sizeof(double) + disp[i-1];
    }
    int result =  MPI_Type_create_struct(20,blocklen,disp,types,
                                         MPI_CONFIG);
    if(result == MPI_SUCCESS)
    {
        if (MPI_Type_commit(MPI_CONFIG) != MPI_SUCCESS)
        {
            printf("ERROR: MPI commit config error!\n");
            exit(0);
        }
    }
    else
    {
        printf("ERROR: MPI create configuration error!\n");
        exit(0);
    }
    printf("done!\n");
}



void MPI_Create_Sp_3Matrix(int len, MPI_Datatype * MPI_SP_3MATRIX){
    printf("creating sp_3matrix\n");
    MPI_Datatype types[4] = {MPI_UNSIGNED,MPI_UNSIGNED,MPI_UNSIGNED,MPI_FLOAT};
    sp_3matrix *tmp = sp_3matrix_alloc(len,len,len);
    int blocklen[4]={1,1,1, len*len*len};
    MPI_Aint disp[4];
    disp[0] = 0;
    disp[1] = sizeof( tmp->x);
    disp[2] = sizeof(tmp->y) + sizeof(tmp->x);
    disp[3] = sizeof(tmp->z) + sizeof(tmp->y) + sizeof(tmp->x);
    int result =  MPI_Type_create_struct(4,blocklen,disp,types,
                                         MPI_SP_3MATRIX);
    if(result == MPI_SUCCESS)
    {
        if (MPI_Type_commit(MPI_SP_3MATRIX) != MPI_SUCCESS)
        {
            printf("ERROR: MPI commit SP_3MATRIX error!\n");
            exit(0);
        }
    }
    else
    {
        printf("ERROR: MPI create SP_3MATRIX error!\n");
        exit(0);
    }
    printf("done!\n");
}

void MPI_Create_Sp_Imatrix(int len,MPI_Datatype* MPI_SP_IMATRIX){
    sp_imatrix *mask = sp_imatrix_alloc(len,len);
    printf("creating sp_imatrix \n");
    MPI_Datatype types[3] = {MPI_UNSIGNED,MPI_UNSIGNED,MPI_INT};
    int blocklen[3]={1,1, len*len};
    MPI_Aint disp[3];
    disp[0] = 0;
    disp[1] = sizeof( mask->rows);
    disp[2] = sizeof(mask->rows) + sizeof(mask->cols);
    int result =  MPI_Type_create_struct(3,blocklen,disp,types,
                                        MPI_SP_IMATRIX);
    if(result == MPI_SUCCESS)
    {
        if (MPI_Type_commit(MPI_SP_IMATRIX) != MPI_SUCCESS)
        {
            printf("ERROR: MPI commit SP_IMATRIX error!\n");
            exit(0);
        }
    }
    else
    {
        printf("ERROR: MPI create SP_IMATRIX error!\n");
        exit(0);
    }
    printf("done!\n");
}

void MPI_Create_Sp_Matrix(int len,MPI_Datatype* MPI_SP_MATRIX ){
    sp_matrix *mask = (sp_matrix*)sp_matrix_alloc(len,len);
    printf("creating sp_matrix\n");
    MPI_Datatype types[3] = {MPI_UNSIGNED,MPI_UNSIGNED,MPI_FLOAT};
    int blocklen[3]={1,1, len*len};
    MPI_Aint disp[3];
    disp[0] = 0;
    disp[1] = sizeof( mask->rows);
    disp[2] = sizeof(mask->rows) + sizeof(mask->cols);
    int result =  MPI_Type_create_struct(3,blocklen,disp,types,
                                        MPI_SP_MATRIX);
    if(result == MPI_SUCCESS)
    {
        if (MPI_Type_commit(MPI_SP_MATRIX)!= MPI_SUCCESS)
        {
            printf("ERROR: MPI commit SP_MATRIX error!\n");
            exit(0);
        }
    }
    else
    {
        printf("ERROR: MPI create SP_MATRIX error!\n");
        exit(0);
    }
    printf("done!\n");
}

/*
void MPIHelper::NewAllTypes(int len){
    Create_Sp_3Matrix(len, );
    Create_Sp_Imatrix(len);
    Create_Sp_Matrix(len);
}
*/

void MPI_Broadcast_Coordinate(sp_matrix* coor, int root){
    real * tmp = coor->data;
    coor->data = NULL;
    // int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
    //    int root, MPI_Comm comm)
    MPI_Bcast(&(coor->rows),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    //    COMM_WORLD.Bcast(&(coor->rows),1,UNSIGNED,root);
    //  COMM_WORLD.Bcast(&(coor->cols),1,UNSIGNED,root);
    MPI_Bcast(&(coor->cols),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    //this->NewAllTypes(coor->rows);
    MPI_Bcast(tmp,coor->rows*coor->cols,MPI_FLOAT,root,MPI_COMM_WORLD);
    // COMM_WORLD.Bcast(tmp, coor->rows*coor->cols, FLOAT, root);
    coor->data = tmp;
}

void MPI_Broadcast_Images(sp_matrix ** images, int N_images,int root){
    real* tmp;
    for(int i = 0; i<N_images; i++){
        tmp = images[i]->data;
        images[i]->data = NULL;
        MPI_Bcast(&(images[i]->rows),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
        MPI_Bcast(&(images[i]->cols),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
        MPI_Bcast(tmp, (images[i]->rows)*(images[i]->cols),MPI_FLOAT, root,
                   MPI_COMM_WORLD);
        images[i]->data = tmp;
    }
    printf("broadcasting images done!\n");
}




void MPI_Broadcast_Mask(sp_imatrix * mask, int root){
    int* tmp = mask->data;
    printf("broadcasting mask!");
    mask->data = NULL;
    MPI_Bcast(&(mask->rows),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    MPI_Bcast(&(mask->cols),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    //COMM_WORLD.Bcast(mask, 1,MPI_SP_MATRIX, taskid);
    MPI_Bcast(tmp, (mask->rows)*(mask->cols),MPI_INT, root, MPI_COMM_WORLD);
    mask->data = tmp;
}


void MPI_Broadcast_Masks(sp_imatrix** masks, int N_images, int root){
    int* tmp;
    for(int i = 0; i<N_images; i++){
        tmp = masks[i]->data;
        masks[i]->data = NULL;
        MPI_Bcast(&(masks[i]->rows),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
        MPI_Bcast(&(masks[i]->cols),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
        //COMM_WORLD.Bcast(masks[i],1,MPI_SP_MATRIX,taskid);
        MPI_Bcast(tmp, (masks[i]->rows)*(masks[i]->cols), MPI_INT, root,
                  MPI_COMM_WORLD);
        masks[i]->data = tmp;
    }
    printf("broadcasting masks done!\n");
}

void MPI_Broadcast_Model(sp_3matrix *model, int taskid){
    printf("broadcasting model\n");
    MPI_Broadcast_3matrix(model, taskid);
}


void MPI_Broadcast_Weight(sp_3matrix* weight, int taskid){
    printf("broadcasting weight\n");
    MPI_Broadcast_3matrix(weight, taskid);
}

void MPI_Broadcast_real(real* vector, int len, int taskid){
    printf("Broadcast real ing...");
    MPI_Bcast(vector,len,MPI_FLOAT, taskid,MPI_COMM_WORLD);
    printf("done!\n");
}


void MPI_Broadcast_3matrix(sp_3matrix *ma, int root){
    real * tmp = ma->data;
    ma->data = NULL;
    MPI_Bcast(&(ma->x),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    MPI_Bcast(&(ma->y),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    MPI_Bcast(&(ma->z),1,MPI_UNSIGNED,root,MPI_COMM_WORLD);
    // COMM_WORLD.Bcast(ma, 1,MPI_SP_3MATRIX, taskid);
    MPI_Bcast(tmp, (ma->x)*(ma->y)*(ma->z),MPI_INT, root,MPI_COMM_WORLD);
    ma->data = tmp;
}


void MPI_Broadcast_Config(ConfigD* conf, int taskid, MPI_Datatype * MPI_CONFIG){
    printf("broadcasting config!...");
    MPI_Bcast(conf, 1,*MPI_CONFIG, taskid,MPI_COMM_WORLD);
    printf("done!\n");
}

void MPI_Send_Model(sp_3matrix* model, int taskid, int flag){
    MPI_Send_3matrix(model,taskid,flag);
}

//send from slaves back to master
void MPI_Send_3matrix(sp_3matrix * ma, int taskid, int flag){
    real * tmp = ma->data;
    ma->data = NULL;
    MPI_Send(tmp, (ma->x)*(ma->y)*(ma->z),MPI_INT, taskid,flag,MPI_COMM_WORLD);
    ma->data = tmp;
}

int  MPI_Recv_3matrix(int len, int root,int flag, sp_3matrix* ma){
    //sp_3matrix * ma= sp_3matrix_alloc(len,len,len);
    MPI_Status st;
    ma->data = NULL;
    ma->x = len;
    ma->y = len;
    ma->z = len;
    printf("receiving 3matrix ...");
    real * tmp=(real*) malloc(sizeof(real)*len*len*len);
    MPI_Recv(tmp,len*len*len,MPI_INT, MPI_ANY_SOURCE,0 ,MPI_COMM_WORLD,&st);
    ma->data = tmp;
    printf("done ! from slave %d\n", st.MPI_SOURCE);
    return st.MPI_SOURCE;
}

void MPI_Send_Respons_unsorted(real* res,int master, int len){
    printf("sending respons...");
    MPI_Send(res, len,MPI_FLOAT, master,9,MPI_COMM_WORLD);
    printf("done!\n");
}

void MPI_Send_Respons(real* res,int master, int len, int rank){
    printf("sending respons...");
    MPI_Send(&rank,1,MPI_INT,master,9,MPI_COMM_WORLD);
    MPI_Send(res, len,MPI_FLOAT, master,9,MPI_COMM_WORLD);
    printf("done!\n");
}

void MPI_Send_Real(real* res,int master, int len,int a){
    printf("sending real...");
    MPI_Send(res, len,MPI_FLOAT, master,a,MPI_COMM_WORLD);
    printf("done!\n");
}

void MPI_Recv_Respons_unsorted(real* res, int len, int N_images, int offset){
    //real* tmp = (real*) malloc(sizeof(real)*len);
    real * tmp=(real*) malloc(sizeof(real)*len * N_images);
    MPI_Status st;
    MPI_Recv(tmp,len*N_images, MPI_FLOAT, MPI_ANY_SOURCE,9 ,MPI_COMM_WORLD,&st);
    // printf("done!\n");
    //printf("copying ...");
    for(int i = 0; i<len; i++)
        res[i + offset] = tmp[i];
    //printf("   Recieveing respons %f %f %f %f %f %f \n", res[0], res[197], tmp[0],tmp[197], res[offset],res[offset+197]  );
    //printf("done!\n");
}

real* MPI_Recv_Respons( int* lens, int N_images, int * rank){
    MPI_Status st;
    MPI_Recv(rank,1, MPI_INT, MPI_ANY_SOURCE,9,MPI_COMM_WORLD,&st);
    real* tmp = (real*) malloc(sizeof(real)*lens[*rank]*N_images);
    MPI_Recv(tmp,lens[*rank]*N_images, MPI_FLOAT, MPI_ANY_SOURCE,9,MPI_COMM_WORLD,&st);
    return tmp;
}

/*
real* MPI_Recv_Respons(int len, int N_images){
    //real* tmp = (real*) malloc(sizeof(real)*len);
    real * tmp=(real*) malloc(sizeof(real)*len * N_images);
    MPI_Status st;
    MPI_Recv(tmp,len*N_images, MPI_FLOAT, MPI_ANY_SOURCE,9,MPI_COMM_WORLD,&st);
    return tmp;
}

void MPI_Recv_Respons(int len, int N_images,real* tmp){
    //real* tmp = (real*) malloc(sizeof(real)*len);
    //real * tmp=(real*) malloc(sizeof(real)*len * N_images);
    MPI_Status st;
    MPI_Recv(tmp,len*N_images,MPI_FLOAT, MPI_ANY_SOURCE,9,MPI_COMM_WORLD,&st);
    //printf("   Recieveing respons %f %f  \n", tmp[43260* N_images-2],tmp[43260* N_images-3]);
}
*/

void MPI_Recv_Real(int len, int N_images,int a, real* returnTmp){
    //real* tmp = (real*) malloc(sizeof(real)*len);
    //real * tmp=(real*) malloc(sizeof(real)*len * N_images);
    MPI_Status st;
    MPI_Recv(returnTmp,len*N_images, MPI_FLOAT, MPI_ANY_SOURCE,a,
             MPI_COMM_WORLD,&st);
    //return tmp;
}

void MPI_Global_Allreduce(real* sendbuf, real* outbuf, int count, MPI_Op op ){
//int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
//    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    int error = MPI_Allreduce((void*)sendbuf,(void*)outbuf,count,MPI_EMC_PRECISION,op,MPI_COMM_WORLD);
    if (error == MPI_SUCCESS)
        printf("ALL REDUCE SUCCESS count %d\n", count);
    else
        printf("all reduce error: %d\n",error);
}

void MPI_Global_Allreduce_com(real* sendbuf, real* outbuf, int count, MPI_Op op, MPI_Comm communicator){
//int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
//    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    int error = MPI_Allreduce((void*)sendbuf,(void*)outbuf,count,MPI_EMC_PRECISION,op,communicator);
    if (error == MPI_SUCCESS)
        printf("ALL REDUCE SUCCESS count %d\n", count);
    else
        printf("all reduce error: %d\n",error);
}

void MPI_Global_Allgather(real* sendbuf, int sendcount,real* recvbuf, int recvcount,int root){
    printf("All Gather %d %d\n",sendcount,recvcount);
    int error =MPI_Gather((void*)sendbuf, sendcount,MPI_EMC_PRECISION
                             ,(void*)recvbuf,recvcount,MPI_EMC_PRECISION,
                          root,MPI_COMM_WORLD);
    if (error == MPI_SUCCESS)
        printf("ALL Gather SUCCESS count %d\n", sendcount);
    else
        printf("All Gather error: %d\n",error);
}
