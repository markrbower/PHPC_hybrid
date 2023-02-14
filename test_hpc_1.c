#include <stdio.h>
#include <sys/time.h>
#include "timer.h"

#define NSIZE 80000000
static double a[NSIZE], b[NSIZE], c[NSIZE];

int main(int argc, char *argv[]) {
  int ntimes=16;
  int sumAll=0;
  double scalar = 3.0, time_sum = 0.0;
  struct timespec tstart;

  for (int i=0; i<NSIZE; i++) {
	a[i] = 1.0;
        b[i] = 2.0;
  }

  cpu_timer_start(&tstart);
  for (int k=0; k<ntimes; k++){
       for (int i=0; i<NSIZE; i++){
           c[i] = a[i] + scalar*b[i];
	   sumAll = sumAll + c[i];
       }
  }
  time_sum = cpu_timer_stop(tstart);

  printf( "%f\n", sumAll );
  printf("Average runtime for stream triad loop is %lf msecs\n", time_sum/ntimes);

}

