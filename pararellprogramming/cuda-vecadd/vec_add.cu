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
  *c = *a + *b;
}

int main() {
  // int N = 16384;    // = 32 * 512
  // int *A = (int *) malloc(N * sizeof(int));
  // int *B = (int *) malloc(N * sizeof(int));
  // int *C = (int *) malloc(N * sizeof(int));
  // int *C_ans = (int *) malloc(N * sizeof(int));

  // for (int i = 0; i < N; i++) {
  //   A[i] = rand() % 1000;
  //   B[i] = rand() % 1000;
  //   C_ans[i] = A[i] + B[i];
  // }

  // TODO: Run vector addition on GPU
  // Save the result in C
  int a = 1, b = 2, c;
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeof(int));
  cudaMalloc(&d_b, sizeof(int));
  cudaMalloc(&d_c, sizeof(int));

  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  kernel_add<<<1,1>>>(d_a, d_b, d_c);
  cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("c : %d \n",c);


  // for (int i = 0; i < N; i++) {
  //   if (C[i] != C_ans[i]) {
  //     printf("Result differ at %d: %d vs %d\n", i, C[i], C_ans[i]);
  //   }
  // }

  printf("Validation done.\n");

  return 0;
}
