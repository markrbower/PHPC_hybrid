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

#define NSIZE 80000000

int main(int argc, char *argv[]) {
  int n_elements_received;
  int np, rank;

  double* restrict a = (double*)malloc(NSIZE * sizeof(double));
  double* restrict a2 = (double*)malloc(NSIZE * sizeof(double));
  void* va = (void*)malloc(NSIZE * sizeof(double));

  omp_set_num_threads(4);

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if ( rank == 0 ) { // Manager
	int index,i;
      	int elements_per_process = NSIZE / np;

	if ( np > 1 ) {
		for (i = 1; i < np; i++) {
                	index = i * elements_per_process;
  
   	       		MPI_Send(&elements_per_process,
                       		1, MPI_INT, i, 0,
                       		MPI_COMM_WORLD);
               		MPI_Send( &a[index],
                       		elements_per_process,
                       		MPI_INT, i, 0,
                       		MPI_COMM_WORLD);
            	}
	} // np > 0	
        printf("Manager done.\n");
  } else { // Worker
	printf("Worker %d started.\n", rank );
        MPI_Recv(&n_elements_received,
                1, MPI_INT, 0, 0,
                MPI_COMM_WORLD,
                &status);
  
	printf("Worker %d middle.\n", rank );
        // stores the received array segment
        // in local array a2
        MPI_Recv( va, n_elements_received,
               	MPI_DOUBLE, 0, 0,
               	MPI_COMM_WORLD,
               	&status);

	a2 = (double*) va;
	printf("Worker %d done.\n", rank );
  }

  free(a);
  free(a2);

  MPI_Finalize();
  return(0);
}


