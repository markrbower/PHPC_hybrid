#include <stdio.h>
#include <omp.h>
#include <mpi.h>

// mpic++ -acc -ta=tesla:managed -Minfo=accel -fopenmp -mcmodel=medium hpc_add.c -o hpc_add
// mpirun hpc_add

#define NSIZE 100000000
#define NUM_THREADS 4

int main(int argc, char ** argv)
{
    int pid, np;

    int provided;

    int* restrict a = (int*)malloc(NSIZE * sizeof(int));

    double start_time;

    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    int nbr_lanes = np * NUM_THREADS;
    printf( "%d\n", nbr_lanes );
    int sum = 0;
    int chunk = NSIZE / (nbr_lanes);

    MPI_Barrier(MPI_COMM_WORLD);
    if ( pid == 0 ) {
        #pragma omp master
        if ( provided != MPI_THREAD_FUNNELED ) {
             printf( "MPI_THREAD_FUNNELED not available. Aborting.\n" );
             MPI_Finalize();
             exit(0);
        }
        printf("requesting MPI_THREAD_FUNNELED with %d threads.\n", omp_get_num_threads());
	for ( int i=0; i<NSIZE; i++ ) {
		a[i] = 1;
	}
    }
    omp_set_num_threads( NUM_THREADS );
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction (+:sum) schedule(static,chunk)
    for ( int i=0; i<NSIZE; i++ ) {
	sum += a[i];
    }
    double duration = omp_get_wtime() - start_time;
    printf( "%d\n", sum );
    printf( "%f\n", duration );

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

