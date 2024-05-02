#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vec_add_kernel(const int *A, const int *B, int *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
}

__global__ void kernel_add(const int* a, const int* b, int* c){
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  c[globalIdx] = a[globalIdx] + b[globalIdx];
  return;
}

int main() {
  int N = 16384;     // = 32 * 512
  // int blkSize = 32;  // thread groups execute the same kernel code together
  int *A = (int *) malloc(N * sizeof(int));
  int *B = (int *) malloc(N * sizeof(int));
  int *C = (int *) malloc(N * sizeof(int));
  int *C_ans = (int *) malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    A[i] = rand() % 1000;
    B[i] = rand() % 1000;
    C_ans[i] = A[i] + B[i];
  }

  // device pointers for a,b,c
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeof(int) * N);
  cudaMalloc(&d_b, sizeof(int) * N);
  cudaMalloc(&d_c, sizeof(int) * N);
  
  cudaMemcpy(d_a, A, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, sizeof(int) * N, cudaMemcpyHostToDevice);
  int blkSize = 32;
  dim3 blkDim(blkSize);
  dim3 gridDim((N + blkSize - 1) / blkSize); // = 512 -> collection of blocks
    
  kernel_add<<<gridDim, blkDim>>>(d_a, d_b, d_c);
  cudaMemcpy(C, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    if (C[i] != C_ans[i]) {
      printf("Result differ at %d: %d vs %d\n", i, C[i], C_ans[i]);
    }
  }

  printf("Validation done.\n");

  return 0;
}
