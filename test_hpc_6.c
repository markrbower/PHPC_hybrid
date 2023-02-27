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
double compute_sum(double* restrict array, int num_elements) {
    double sum = 0.0;
    int i;
    #pragma acc parallel loop gang device_type(acc_device_nvidia) vector_length(256)
    for (i = 0; i < num_elements; i++) {
      sum += array[i];
    }
    return sum;
}

// test_hpc_6
//
// Can using Scatter/Gather speed things up?
// 
int main(int argc, char *argv[]) {
  struct timespec tstart;
  double time_all = 0.0;
//  double scalar = 3.0;
  int np, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  double* restrict a = (double*)malloc(NSIZE * sizeof(double));
  double* restrict b = (double*)malloc(NSIZE * sizeof(double));
  double* restrict c = (double*)malloc(NSIZE * sizeof(double));

  void* va = (void*)malloc(NSIZE*sizeof(double));
  void* vb = (void*)malloc(NSIZE*sizeof(double));

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  int elements_per_proc = NSIZE / np;

  printf( "Running on %d processors.\n", np );
       if ( rank == 0 ) { // Manager
  		#pragma acc parallel loop present( a[0:NSIZE], b[0:NSIZE], c[0:NSIZE] )
  		for (int i=0; i<NSIZE; i++) {
			a[i] = 1.0;
       			b[i] = 2.0;
  		}
	}
        // Create a buffer that will hold a subset
	double* restrict sub_array = (double*) malloc(sizeof(double) * elements_per_proc);

     	cpu_timer_start(&tstart);
	// Scatter the random numbers to all processes
	MPI_Scatter(a, elements_per_proc, MPI_DOUBLE,
		sub_array, elements_per_proc, MPI_DOUBLE,
		0, MPI_COMM_WORLD);

	// Compute the average of your subset
	double sub_sum = compute_sum(sub_array, elements_per_proc);

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
    free(sub_array);
    free(sub_sums);
  }
  free(sub_array);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

}


