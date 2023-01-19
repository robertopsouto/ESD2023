#include <stdio.h>
#include <assert.h>

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

template <typename T>
__global__ void strideCopy(T* a, int stride)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
  a[i] = a[i] + 1;
}

template <typename T>
__global__ void offsetCopy(T* a, int offset)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 256;
  float ms;

  T *d_a;
  cudaEvent_t startEvent, stopEvent;
    
  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB) for stride case
  checkCuda( cudaMalloc(&d_a, n * 33 * sizeof(T)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("Offset, Bandwidth (GB/s):\n");
  
  offsetCopy<<<n/blockSize, blockSize>>>(d_a, 0); // warm up

  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0.0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    offsetCopy<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");

  strideCopy<<<n/blockSize, blockSize>>>(d_a, 1); // warm up
  for (int i = 1; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0.0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    strideCopy<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
      fprintf(stderr, "Syntax: %s <device id>\n", argv[0]);
      return EXIT_FAILURE;
  }

  int deviceId = atoi(argv[1]);
  checkCuda( cudaSetDevice(deviceId) );

  int nMB = 4;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {    
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }
  
  cudaDeviceProp prop;
  
  checkCuda( cudaSetDevice(deviceId) );
  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);
  
  printf("%s Precision\n", bFp64 ? "Double" : "Single");
  
  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}
