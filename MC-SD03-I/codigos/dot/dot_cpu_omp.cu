#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

void dot( float *a, float *b, float &c, int N ) {
    //int tid;    // this is CPU zero, so we start at zero
    #pragma omp parallel for reduction(+:c)
    for (int tid = 0; tid < N; tid++) { 
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
    float  *a, *b; 
    // allocate memory on the cpu side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    // fill in the host memory with data
    size_t i;
    for (i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    float c = 0.0f;

    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    dot( a, b, c, N );
    gettimeofday(&end, NULL);

    float cpuTime = 1000000*(float)(end.tv_sec - begin.tv_sec);
    cpuTime +=  (float)(end.tv_usec - begin.tv_usec);

    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf( "Does CPU value %.6g = %.6g?\n", c, 2 * sum_squares( (float)(N - 1) ) );
    // print times
    printf("\nExecution Time (microseconds): %9.2f\n", cpuTime);

    // free the memory allocated on the CPU
    free(a); a=NULL;
    free(b); b=NULL;

}
