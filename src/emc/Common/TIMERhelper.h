#ifndef TIMER_HELPER_H
#define TIMER_HELPER_H
#include <time.h>
#include <sys/time.h>
unsigned long int gettimenow();
double update_time(unsigned long int before, unsigned long int now);
#endif // TIMER_HELPER_H
