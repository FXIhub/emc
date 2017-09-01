#include "TIMERhelper.h"
#include <stdlib.h>

unsigned long int gettimenow(){
    struct timeval now;
    gettimeofday(&now,NULL);
    return  now.tv_sec*1000+now.tv_usec/1000;
}

double update_time( unsigned long int before, unsigned long int nowi){
    return  (nowi - before) / 1000.0;
}
