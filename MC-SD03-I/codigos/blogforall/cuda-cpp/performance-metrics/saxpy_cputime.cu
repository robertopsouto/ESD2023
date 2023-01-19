#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char* argv[])
{

  if (argc != 3) {
    fprintf(stderr, "Syntax: %s <matrix size N> <block size> <device id>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int N = atoi(argv[1]);
  int devId = atoi(argv[2]);

  printf("Number of Elements : %d\n", N);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devId);
  printf("Device : %s\n", prop.name);
  cudaSetDevice(devId);

  float *x, *y;
  // allocate the memory on the CPU
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  float *d_x, *d_y;
  // allocate the memory on the GPU
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  
  // Perform SAXPY on 1M elements
  saxpy<<<(N+511)/512, 512>>>(N, 2.0f, d_x, d_y);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  double cpuTime = 1000000*(double)(end.tv_sec - begin.tv_sec);
  cpuTime +=  (double)(end.tv_usec - begin.tv_usec);
  
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }

  printf("Max error: %f\n", maxError);
  printf("Execution Time (miliseconds): %f\n", cpuTime/1000.0);
  printf("Effective Bandwidth (GB/s): %f\n", N*4*3/cpuTime*1e6/1e9);
  printf("Effective Performance (GFLOP/s): %f\n", N*2/cpuTime*1e6/1e9);

  // free the memory allocated on the CPU
  free(x); x=NULL;
  free(y); y=NULL;

  // free the memory allocated on the GPU
  cudaFree( d_x );
  cudaFree( d_y );
}
