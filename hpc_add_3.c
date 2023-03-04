#include <stdio.h>
#include <omp.h>
#include <mpi.h>

// mpic++ -fopenmp mpi_omp_hello.c -o mpi_omp_hello
// mpirun mpi_omp_hello

int main(int argc, char ** argv)
{
    int pid, np;

    int provided;

    int sum[20][8] = {0};

    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    printf("Sequential %d out of %d!\n",pid,np);
    MPI_Barrier(MPI_COMM_WORLD);
    if ( pid == 0 ) {
        #pragma omp master
        if ( provided != MPI_THREAD_FUNNELED ) {
             printf( "MPI_THREAD_FUNNELED not available. Aborting.\n" );
             MPI_Finalize();
             exit(0);
        }
    }
    omp_set_num_threads(4);
    printf("requesting MPI_THREAD_FUNNELED with %d threads.\n", omp_get_num_threads());
    #pragma omp parallel
    	{
        	int thnum = omp_get_thread_num();
        	int thtotal = omp_get_num_threads();
		int my_num = pid*thtotal + thnum;
        	printf("parallel: %d out of %d threads from proc %d out of %d gives %d\n",thnum,thtotal,pid,np,my_num);
		sum[my_num][0] = 1;
    	}
    MPI_Barrier(MPI_COMM_WORLD);
    int total = 0;
    if  ( pid == 0 ) {
	for ( int j=0; j<20; j++ ) {
	    	total = total + sum[j][0];
	}
	printf( "total: %d\n", total );
    }	


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
