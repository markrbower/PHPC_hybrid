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

// Computes the average of an array of numbers
double compute_scaled_sum(double* restrict a, double* restrict b, double scalar, int num_elements) {
    double sum = 0.0;
    int i;
    #pragma acc parallel loop gang device_type(acc_device_nvidia) vector_length(256)
    for (i = 0; i < num_elements; i++) {
      sum += a[i] + scalar*b[i];
    }
    return sum;
}

// Computes the scale and sum of two arrays of numbers
double compute_sum(double* restrict array, int num_elements) {
    double sum = 0.0;
    int i;
    #pragma acc parallel loop gang device_type(acc_device_nvidia) vector_length(256)
    for (i = 0; i < num_elements; i++) {
      sum += array[i];
    }
    return sum;
}

// test_hpc_7.c
//
// Can using Scatter/Gather speed things up?
// 
int main(int argc, char *argv[]) {
  struct timespec tstart;
  double time_all = 0.0;
  double scalar = 3.0;
  int np, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  double* restrict a = (double*)malloc(NSIZE * sizeof(double));
  double* restrict b = (double*)malloc(NSIZE * sizeof(double));
  double* restrict c = (double*)malloc(NSIZE * sizeof(double));

  void* va = (void*)malloc(NSIZE*sizeof(double));
  void* vb = (void*)malloc(NSIZE*sizeof(double));

  int provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  int elements_per_proc = NSIZE / np;

  printf( "Running on %d processors.\n", np );
  if ( rank == 0 ) { // Manager
	#pragma omp parallel
	printf( "requesting MPI_THREAD_FUNNELED"
			" with %d threads\n",
			omp_get_num_threads());
	if ( provided != MPI_THREAD_FUNNELED ) {
		printf( "Error" );
		MPI_Finalize();
		exit(0);
	}

  	#pragma acc parallel loop present( a[0:NSIZE], b[0:NSIZE], c[0:NSIZE] )
  	for (int i=0; i<NSIZE; i++) {
		a[i] = 1.0;
       		b[i] = 2.0;
  	}
  }
  // Create a buffer that will hold a subset
  double* restrict sub_array_a = (double*) malloc(sizeof(double) * elements_per_proc);
  double* restrict sub_array_b = (double*) malloc(sizeof(double) * elements_per_proc);

  cpu_timer_start(&tstart);
  // Scatter the random numbers to all processes
  MPI_Scatter(a, elements_per_proc, MPI_DOUBLE,
	sub_array_a, elements_per_proc, MPI_DOUBLE,
	0, MPI_COMM_WORLD);
  MPI_Scatter(b, elements_per_proc, MPI_DOUBLE,
	sub_array_b, elements_per_proc, MPI_DOUBLE,
	0, MPI_COMM_WORLD);

  // Compute the average of your subset
  double sub_sum = compute_scaled_sum(sub_array_a, sub_array_b, scalar, elements_per_proc);

  // Gather all partial averages down to the root process
  double* sub_sums = NULL;
  if (rank == 0) {
  	sub_sums = (double*)malloc(sizeof(double) * np);
  }
  MPI_Gather(&sub_sum, 1, MPI_DOUBLE, sub_sums, 1, MPI_DOUBLE, 0,
  MPI_COMM_WORLD);

  // Compute the total average of all numbers.
  if (rank == 0) {
  	double total = compute_sum(sub_sums, np);
  	printf("sum of all elements is %f\n", total );
  	time_all = cpu_timer_stop(tstart);
  	printf("Average runtime for the program is %lf msecs\n", time_all );
  }

  // Clean up
  if (rank == 0) {
   free(a);
   free(b);
   free(sub_array_a);
   free(sub_array_b);
   free(sub_sums);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}


