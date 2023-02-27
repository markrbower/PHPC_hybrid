#include <iostream>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <openacc.h>
#include <time.h>
#include "timer.h"

#define NSIZE 200000000

int main(int argc, char *argv[]) {
  int ntimes=16;
  double sumAll=0.0;
  struct timespec tstart;
  double scalar = 3.0, time_sum = 0.0;

  double* restrict a = (double*)malloc(NSIZE * sizeof(double));
  double* restrict b = (double*)malloc(NSIZE * sizeof(double));
  double* restrict c = (double*)malloc(NSIZE * sizeof(double));

  omp_set_num_threads(4);

  #pragma acc parallel loop present( a[0:NSIZE], b[0:NSIZE], c[0:NSIZE] )
  for (int i=0; i<NSIZE; i++) {
	a[i] = 1.0;
       	b[i] = 2.0;
  }

  cpu_timer_start(&tstart);
  for (int k=0; k<ntimes; k++){
       sumAll = 0.0;
       #pragma acc parallel loop gang device_type(acc_device_nvidia) vector_length(256)
       for (int i=0; i<NSIZE; i++){
           c[i] = a[i] + scalar*b[i];
	   sumAll += c[i];
       }
  }
  time_sum = cpu_timer_stop(tstart);

  printf("Sum is %f.\n", sumAll );
  printf("Average runtime for stream triad loop is %lf msecs\n", time_sum/ntimes);

}

