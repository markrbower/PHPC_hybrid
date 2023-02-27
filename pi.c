//  gcc -fopenmp pi.c -o pi -Wall
//  ./pi
#include <stdio.h>
#include <omp.h>

double pi_r (double h, unsigned depth, unsigned maxdepth, unsigned long long begin, unsigned long long niters)
{
    if (depth < maxdepth) {
	        double area1, area2;

	        // Process first half
		#pragma omp task shared(area1)
		area1 = pi_r (h, depth+1, maxdepth, begin, niters/2-1);
		
		// Process second half
		#pragma omp task shared(area2)
		area2 = pi_r (h, depth+1, maxdepth, begin+niters/2, niters/2);
		
	        #pragma omp taskwait
                return area1+area2;
     } else {
		unsigned long long i;
		double area = 0.0;
                for (i = begin; i <= begin+niters; i++) {
			double x = h * (i - 0.5);
	        	area += (4.0 / (1.0 + x*x));
		}
	        return area;
     }
}

double pi (unsigned long long niters) {
     double res;
     double h = 1.0 / (double) niters;

     #pragma omp parallel shared(res)
     {
     	#define MAX_PARALLEL_RECURSIVE_LEVEL 12

        #pragma omp single
        res = pi_r (h, 0, MAX_PARALLEL_RECURSIVE_LEVEL, 1, niters);
     }
     return res * h;
}

int main (int argc, char *argv[])
{
	long NITERS = 100*1000*1000;

	double start_time = omp_get_wtime();

	printf ("PI (%ld iters) is %lf\n", NITERS, pi(NITERS));

	double run_time = omp_get_wtime() - start_time;
	printf("run time: %f\n", run_time );

        return 0;
}

