#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

int main() {
  // TODO
  int count;
  cudeGetDeviceCount(&count);

  cudaDeviceProp prop[4];
  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop[i], i);
    printf("\t\tname: %s\n", props[i].name);
    printf("\t\tmultiProcessorCount: %d\n", props[i].multiProcessorCount);
    printf("\t\tmaxThreadsPerBlock: %d\n", props[i].maxThreadsPerBlock);
    printf("\t\ttotalGlobalMem: %lu\n", props[i].totalGlobalMem);
    printf("\t\tsharedMemPerBlock: %lu\n", props[i].sharedMemPerBlock);
  
  }
  



  return 0;
}
