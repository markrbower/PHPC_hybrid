#include <stdio.h>
#include <omp.h>

// g++ -fopenmp omp_hello.c -o omp_hello
// ./omp_hello

int main( int argc, char *argv[] ) {
	omp_set_num_threads(4);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int np = omp_get_num_threads();

		printf( "Hello world from %d of %d\n", id, np );
	}
	return(0);
}

