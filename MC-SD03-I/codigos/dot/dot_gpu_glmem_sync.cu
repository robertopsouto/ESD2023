/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


__global__ void dot_prod( float *a, float *b, float *c, int N ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) c[tid] = a[tid] * b[tid];
}

__global__ void dot_reduction( float *c, int i ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid < i ) c[tid] += c[tid + i];
}


int main(int argc, char* argv[])
{

    if (argc != 4) {
      fprintf(stderr, "Syntax: %s <vector size N> <block size> <device id>\n", argv[0]);
      return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int BlockSize = atoi(argv[2]);
    int devId = atoi(argv[3]);

    checkCuda( cudaSetDevice(devId) );
  
    printf("Size of vector: %d \n", N);

    float   *a, *b, *c, r;
    float   *dev_a, *dev_b, *dev_c;

    // allocate memory on the cpu side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    c = (float*)malloc( N*sizeof(float) );

    // allocate the memory on the GPU
    checkCuda( cudaMalloc( (void**)&dev_a, N*sizeof(float) ) );
    checkCuda( cudaMalloc( (void**)&dev_b, N*sizeof(float) ) );
    checkCuda( cudaMalloc( (void**)&dev_c, N*sizeof(float) ) );

    // fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // copy the arrays 'a' and 'b' to the GPU
    checkCuda( cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice ) );
    checkCuda( cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice ) ); 

    // perform product for all pair of elements
    int GridSize =  ( N + BlockSize-1 ) / BlockSize ;
    dot_prod<<<GridSize,BlockSize>>>( dev_a, dev_b, dev_c, N );

    // for reductions, vector size N must be a power of 2
    // because of the following code
    int i = N/2;
    printf("i: %d\n",i);
    while (i != 0) {
        dot_reduction<<<GridSize,BlockSize>>>( dev_c, i );
        checkCuda( cudaDeviceSynchronize() );
        i /= 2;
        printf("i: %d\n",i);
    }

    checkCuda( cudaMemcpy( c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost ) );
    r = c[0]; 

    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf( "Does GPU value %.6g = %.6g?\n", r, 2 * sum_squares( (float)(N - 1) ) );

    // free memory on the gpu side
    checkCuda( cudaFree( dev_a ) );
    checkCuda( cudaFree( dev_b ) );
    checkCuda( cudaFree( dev_c ) );

    // free memory on the cpu side
    free( a );
    free( b );
    free( c );
}
