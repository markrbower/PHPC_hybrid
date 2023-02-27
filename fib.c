#include <stdio.h>
#include <sys/time.h>
#include <omp.h>


int fib(int n)
{
    if(n == 0 || n == 1)
            return n;

    int res, a, b;
    {
            {
                #pragma omp task shared(a)
                a = fib(n-1);
                #pragma omp task shared(b)
                b = fib(n-2);
		#pragma omp taskwait
                res = a+b;
	    } 

    }
    return res;
}

int main()
{  
    struct timeval start, end;
    gettimeofday(&start, NULL);

    int result = 0;
    float delta = 0.0;
    omp_set_num_threads(4);
    #pragma omp parallel 
    {
	    result = fib(40);
    }
    // benchmark code
    gettimeofday(&end, NULL);

    delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
          end.tv_usec - start.tv_usec) / 1.e6;

    printf("answer: %d\ttime: %f\n", result, delta );    
}
