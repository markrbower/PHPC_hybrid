#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <openacc.h>
#include <time.h>
#include "timer.h"

#define NSIZE 200000000
static double a[NSIZE], b[NSIZE], c[NSIZE];

int main(int argc, char *argv[]) {
  int ntimes=16;
  double sumAll=0.0;
  double scalar = 3.0, time_sum = 0.0;
  struct timespec tstart;

  int provided;


  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Barrier(MPI_COMM_WORLD);

  if ( rank == 0 ) {
	#pragma omp master
	if ( provided != MPI_THREAD_FUNNELED ) {
		  printf( "MPI_THREAD_FUNNELED not available. Aborting.\n" );
		  MPI_Finalize();
		  exit(0);
	}
  	omp_set_num_threads(4);
	#pragma omp parallel
	{		
		printf("requesting MPI_THREAD_FUNNELED with %d threads.\n", omp_get_num_threads());
	}
	#pragma omp simd
  	for (int i=0; i<NSIZE; i++) {
        	a[i] = 1.0;
        	b[i] = 2.0;
  	}

  	cpu_timer_start(&tstart);
//	#pragma omp parallel for
  	for (int k=0; k<ntimes; k++){
		sumAll = 0.0;
  		#pragma omp simd reduction(+:sumAll)
       		for (int i=0; i<NSIZE; i++){
           		c[i] = a[i] + scalar*b[i];
           		sumAll = sumAll + c[i];
       		}
  	}
  	time_sum = cpu_timer_stop(tstart);

  	printf("%f\n", sumAll);
  	printf("Average runtime for stream triad loop is %lf msecs\n", time_sum/ntimes);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  exit(0);

}

