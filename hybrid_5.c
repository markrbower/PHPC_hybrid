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
using namespace std;

int main(int argc, char *argv[]) {
	  int numprocs, rank, namelen;
	  char processor_name[MPI_MAX_PROCESSOR_NAME];
	  int iam = 0, np = 1;
	  vector<int> A; 
	  int N=10000;
	  int sumA=0;

	  // ACC stuff here
          int nsize = 20000000, ntimes=16;

          double* a = (double*)malloc(nsize * sizeof(double));
          double* b = (double*)malloc(nsize * sizeof(double));
	  double* c = (double*)malloc(nsize * sizeof(double));

//          #pragma acc enter data create(a[0:nsize],b[0:nsize],c[0:nsize])

//	  #pragma omp target enter data map(to:a[0:nsize], b[0:nsize], c[0:nsize])
	  struct timespec tstart;
          double scalar = 3.0, time_sum = 0.0;
//          #pragma omp target teams distribute parallel for simd
          for (int i=0; i<nsize; i++) {
	  	a[i] = 1.0;
	        b[i] = 2.0;
	  }

	  // MPI stuff here
	  MPI_Init(&argc, &argv);
	  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	  MPI_Get_processor_name(processor_name, &namelen);

	  // OMP
//          #pragma omp parallel default(shared) private(iam, np)
	  {
	    np = omp_get_num_threads();
	    iam = omp_get_thread_num();
	    printf("Hello from thread %d out of %d from process %d out of %d on %s\n",
			           iam, np, rank, numprocs, processor_name);
	  }

	  A.resize(N);
	  std::fill(A.begin(),A.end(),1);
	  double start_time = omp_get_wtime();
//	   #pragma omp parallel for reduction(+ : sumA) 
           //#pragma acc parallel loop reduction(+:sumA)
	   for (int i=0;i<N;i++) {                     
	      for (int j=0;j<N;j++) {
	          sumA += A[i];  // just adding 1 to sum N times
	      }
       //       sumA -= N;   // subtract off N to reset back to zero 
	   }
	   double time = omp_get_wtime() - start_time;
	   if (rank == 0) {
	   	cout<<"   rank  "<< "    sum     "<<"  "<<"time in sec"<<endl;
	   }

	   MPI_Barrier( MPI_COMM_WORLD );
	   cout << rank << "        " << sumA << "       " << time << endl;

	   for (int k=0; k<ntimes; k++){
	       cpu_timer_start(&tstart);
               //#pragma omp target teams distribute parallel for simd
//	       #pragma acc parallel loop
	       for (int i=0; i<nsize; i++){
	           c[i] = a[i] + scalar*b[i];
	       }
	       time_sum += cpu_timer_stop(tstart);
          }

 //         #pragma acc exit data delete(a[0:nsize],b[0:nsize],c[0:nsize])

	   printf("Average runtime for stream triad loop is %lf msecs\n", time_sum/ntimes);
//           #pragma omp target exit data map(from:a[0:nsize], b[0:nsize], c[0:nsize])

	   free(a);
	   free(b);
	   free(c);

	  MPI_Finalize();
}

