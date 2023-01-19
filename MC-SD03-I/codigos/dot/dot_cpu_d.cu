#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void dot( double *a, double *b, double &c, int N ) {
    int tid;    // this is CPU zero, so we start at zero
    for (tid = 0; tid < N; tid++) { 
        c += a[tid] * b[tid];
    }
}

int main(int argc, char* argv[])
{

    if (argc != 2) {
      fprintf(stderr, "Syntax: %s <vector size N>\n", argv[0]);
      return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    double  *a, *b; 
    // allocate memory on the cpu side
    a = (double*)malloc( N*sizeof(double) );
    b = (double*)malloc( N*sizeof(double) );
    // fill in the host memory with data
    size_t i;
    for (i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    double c = 0.0f;

    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    dot( a, b, c, N );
    gettimeofday(&end, NULL);

    double cpuTime = 1000000*(double)(end.tv_sec - begin.tv_sec);
    cpuTime +=  (double)(end.tv_usec - begin.tv_usec);

    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf( "Does CPU value %.6g = %.6g?\n", c, 2 * sum_squares( (double)(N - 1) ) );
    // print times
    printf("\nExecution Time (microseconds): %9.2f\n", cpuTime);

    // free the memory allocated on the CPU
    free(a); a=NULL;
    free(b); b=NULL;

}
