#include <stdio.h>
#include <omp.h>
#include <mpi.h>

// mpic++ -fopenmp mpi_omp_hello.c -o mpi_omp_hello
// mpirun mpi_omp_hello

int main(int argc, char ** argv)
{
    int thnum, thtotal;
    int pid, np;

    int provided;


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
    #pragma omp parallel private(thnum,thtotal)
    	{
        	thnum = omp_get_thread_num();
        	thtotal = omp_get_num_threads();
        	printf("parallel: %d out of %d threads from proc %d out of %d\n",thnum,thtotal,pid,np);
    	}

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
