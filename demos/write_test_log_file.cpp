#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <emc.h>
#include <mpi.h>
#include <MPIhelper.h>
#include <FILEhelper.h>
#include <emc_math.h>
#include <TIMERhelper.h>
using namespace std;
#include <unistd.h>

#include <string.h>
const char* logFile_path ="./logFile.text";
int main(){
    FILE* logFW = fopen(logFile_path,"a+");
    for (int i =0; i<10; i++)
    fprintf(logFW, "%.4d %.8d %.4d %.1d %.4f\n", 3+i,3+i+1,103,1,0.5);
    fclose(logFW);
    return 0;
}
