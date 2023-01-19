#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void saxpy(int n, float a, float *x, float *y)
{
  for (int i = 0; i < n; ++i)
      y[i] = a*x[i] + y[i];
}


int main(int argc, char* argv[])
{
    int N = 1;
    if (argc > 1) N = atoi(argv[1]);
    float *x, *y;
    x = (float *)malloc( N * sizeof (float));
    y = (float *)malloc( N * sizeof (float));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    saxpy( N, 2.0f, x, y );
    gettimeofday(&end, NULL);

    double cpuTime = 1000000*(double)(end.tv_sec - begin.tv_sec);
    cpuTime +=  (double)(end.tv_usec - begin.tv_usec);

    // print times
    printf("Execution Time (milliseconds): %f\n", cpuTime/1e3);
    printf("Effective Bandwidth (GB/s): %f\n", N*4*3/cpuTime/1e3);
    printf("Effective Performance (GFLOP/s): %f\n", N*2/cpuTime/1e3);

    // free the memory allocated on the CPU
    free(x); x=NULL;
    free(y); y=NULL;

    return 0;
}
