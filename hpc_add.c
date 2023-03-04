#include <stdio.h>
#include <omp.h>
#include <mpi.h>

// mpic++ -acc -ta=tesla:managed -Minfo=accel -fopenmp -mcmodel=medium hpc_add.c -o hpc_add
// mpirun hpc_add

#define NSIZE 100000000
#define NUM_THREADS 4
#define CBLK 8

int main(int argc, char ** argv)
{
    int thnum, thtotal;
    int pid, np;

    int provided;

    int* restrict a = (int*)malloc(NSIZE * sizeof(int));

    double start_time;

    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    int nbr_lanes = np * NUM_THREADS;
    int sum[nbr_lanes][CBLK] = {0};

    if ( pid == 0 ) {
        #pragma omp master
        if ( provided != MPI_THREAD_FUNNELED ) {
             printf( "MPI_THREAD_FUNNELED not available. Aborting.\n" );
             MPI_Finalize();
             exit(0);
        }
    	printf( "%d\n", nbr_lanes );
	for ( int i=0; i<NSIZE; i++ ) {
		a[i] = 1;
	}
    }
    MPI_Bcast( a, NSIZE, MPI_INT, 0, MPI_COMM_WORLD ); 
    omp_set_num_threads( NUM_THREADS );
    start_time = omp_get_wtime();
    #pragma omp parallel private(thnum,thtotal) // creates the "thread team"
    {
    	thnum = omp_get_thread_num();
	if ( pid==0 && thnum==0 ) { // The number of threads assigned available only in parallel section.
        	printf("requesting MPI_THREAD_FUNNELED with %d threads.\n", omp_get_num_threads());
	}
       	thtotal = omp_get_num_threads();
	//printf( "%d\n", thtotal );
	int my_nbr = pid * thtotal + thnum;
	//printf( "%d\n", my_nbr );
	int elements_per_thread = NSIZE / (nbr_lanes);
	int istart = my_nbr * elements_per_thread;
	int iend = (my_nbr+1) * elements_per_thread;
	for ( int i=istart; i<iend; i++ ) {
		sum[my_nbr][0] += a[i];
	}
//	printf("%d : %d : %d : %d\n", istart, iend, my_nbr, sum[my_nbr][0] );
//     	  printf("parallel: %d process: %d out of %d threads from proc %d out of %d\n",pid,thnum,thtotal,pid,np);
    }
    if ( pid == 0 ) {
    	    double duration = omp_get_wtime() - start_time;
	    int total = 0;
            for ( int i=0; i<nbr_lanes; i++ ) {
	         total = total + sum[i][0];
	    }
	    printf( "%d\n", total );
	    printf( "%f\n", duration );
    }

    MPI_Finalize();
    return 0;
}
