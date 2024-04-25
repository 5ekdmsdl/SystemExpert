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

int main() {
  int N = 16384;
  int *A = (int *) malloc(N * sizeof(int));
  int *B = (int *) malloc(N * sizeof(int));
  int *C = (int *) malloc(N * sizeof(int));
  int *C_ans = (int *) malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    A[i] = rand() % 1000;
    B[i] = rand() % 1000;
    C_ans[i] = A[i] + B[i];
  }

  int *A_gpu, *B_gpu, *C_gpu;
  CHECK_CUDA(cudaMalloc(&A_gpu, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&B_gpu, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&C_gpu, N * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(A_gpu, A, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, B, N * sizeof(int), cudaMemcpyHostToDevice));

  dim3 gridDim((N + 1024 - 1) / 1024);
  dim3 blockDim(1024);

  vec_add_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C, C_gpu, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (C[i] != C_ans[i]) {
      printf("Result differ at %d: %d vs %d\n", i, C[i], C_ans[i]);
    }
  }

  printf("Validation done.\n");

  return 0;
}
