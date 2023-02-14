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
  int ntimes=16;
  double sumAll=0.0;
  int n_elements_received;
  struct timespec tstart, start_all;
  double time_sum = 0.0;
  double time_all = 0.0;
  double scalar = 3.0;
  int np, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  double* restrict a = (double*)malloc(NSIZE * sizeof(double));
  double* restrict b = (double*)malloc(NSIZE * sizeof(double));
  double* restrict c = (double*)malloc(NSIZE * sizeof(double));

  void* va = (void*)malloc(NSIZE*sizeof(double));
  void* vb = (void*)malloc(NSIZE*sizeof(double));

  #pragma acc parallel loop present( a[0:NSIZE], b[0:NSIZE], c[0:NSIZE] )
  for (int i=0; i<NSIZE; i++) {
	a[i] = 1.0;
       	b[i] = 2.0;
  }

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  MPI_Barrier(MPI_COMM_WORLD);

//  if ( rank == 0 ) {
//  	printf("Starting the clock!\n");
//  }

  printf( "Running on %d processors.\n", np );
  cpu_timer_start(&start_all);
  for (int k=0; k<ntimes; k++ ) {
       if ( rank == 0 ) { // Manager
		int index,i;
  	    	int elements_per_process = NSIZE / np;

		if ( np > 1 ) {
			for (i = 1; i < np - 1; i++) {
	                	index = i * elements_per_process;
  
       	        		MPI_Send(&elements_per_process,
                        		1, MPI_INT, i, 0,
                        		MPI_COMM_WORLD);
                		MPI_Send(&a[index],
                        		elements_per_process,
                        		MPI_INT, i, 0,
                        		MPI_COMM_WORLD);
                		MPI_Send(&b[index],
                        		elements_per_process,
                        		MPI_INT, i, 0,
                        		MPI_COMM_WORLD);
            		}
			// send remaining elements to last processor
            		index = i * elements_per_process;
            		int elements_left = NSIZE - index;
  
            		MPI_Send(&elements_left,
                     		1, MPI_INT,
                     		i, 0,
                     		MPI_COMM_WORLD);
            		MPI_Send(&a[index],
                     		elements_left,
                     		MPI_DOUBLE, i, 0,
                     		MPI_COMM_WORLD);
            		MPI_Send(&b[index],
                     		elements_left,
                     		MPI_DOUBLE, i, 0,
                     		MPI_COMM_WORLD);
		} // np > 0	
		// master process add its own sub array
//		printf("M\n");
		sumAll = 0.0;
       		cpu_timer_start(&tstart);
		#pragma acc parallel loop gang device_type(acc_device_nvidia) vector_length(256)
        	for (int i = 0; i < elements_per_process; i++)
            		sumAll += a[i] + scalar*b[i];

           	time_sum += cpu_timer_stop(tstart);
  
        	// collects partial sums from other processes
		int tmp[2];
        	for (i = 1; i < np; i++) {
//			printf( "Manager receiving %d.\n", i );
            		MPI_Recv(&tmp, 2, MPI_INT,
                     		MPI_ANY_SOURCE, 0,
                     		MPI_COMM_WORLD,
                     		&status);
            		sumAll += tmp[0];
			time_sum += (double) tmp[1]/1E6;
//			printf( "%f\n", time_sum );
        	}
        	// prints the final sum of array
//        	printf("Sum of array is : %f\n", sumAll);
//        	printf("Time sum is : %f\n", time_sum );
       } else { // Worker
        	MPI_Recv(&n_elements_received,
                 	1, MPI_INT, 0, 0,
                 	MPI_COMM_WORLD,
                 	&status);
  
        	// stores the received array segment
        	// in local array a2 & b2
        	MPI_Recv( va, n_elements_received,
                 	MPI_DOUBLE, 0, 0,
                 	MPI_COMM_WORLD,
                 	&status);
        	MPI_Recv( vb, n_elements_received,
                 	MPI_DOUBLE, 0, 0,
                 	MPI_COMM_WORLD,
                 	&status);

        	// calculates its partial sum
        	double partial_sum = 0.0;
       		cpu_timer_start(&tstart);
		
  		double* restrict a2 = (double*)va;
  		double* restrict b2 = (double*)vb;
		#pragma acc parallel loop gang device_type(acc_device_nvidia) vector_length(256)
        	for (int i = 0; i < n_elements_received; i++)
            		partial_sum += a2[i] + scalar*b2[i];

           	double time_part = cpu_timer_stop(tstart);
  
        	// sends the partial sum to the root process
//		printf( "Worker %d sending.\n", rank );
//		printf( "%f\n", time_part );

		int send_array[2];
		send_array[0] = partial_sum;
		send_array[1] = (int) 1E6 * time_part;
		
        	MPI_Send(&send_array, 2, MPI_INT,
                 	0, 0, MPI_COMM_WORLD);
       } // Worker
  }

  if ( rank == 0 ) {
  	printf("Average runtime for stream triad loop is %lf msecs\n", time_sum/(ntimes*np) );
  	time_all = cpu_timer_stop(start_all);
  	printf("Average runtime for the program is %lf msecs\n", time_all/(ntimes) );
  }

  free(a);
  free(b);
  free(c);

  MPI_Finalize();
  return(0);
}


