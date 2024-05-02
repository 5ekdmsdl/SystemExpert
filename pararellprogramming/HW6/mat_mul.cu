#include <cuda_runtime.h>

#include <cstdio>

#include "mat_mul.h"

#define CUDA_CALL(f)                                                       \
  {                                                                        \
    cudaError_t err = (f);                                                 \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__, \
              err, cudaGetErrorString(err));                               \
      exit(1);                                                             \
    }                                                                      \
  }

// Super slow sgemm kernel
__global__ void sgemm(float *A, float *B, float *C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;

  C[i * N + j] = 0;
  for (int k = 0; k < K; ++k) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}

// Device (GPU) pointers
static float *a_d;
static float *b_d;
static float *c_d;

void mat_mul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  dim3 blockDim(1, 1, 1);
  dim3 gridDim(M, N, 1);

  sgemm<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K) {
  // Allocate device memory
  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));

  // Upload A and B matrix to GPU
  CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void mat_mul_final(float *A, float *B, float *C, int M, int N, int K) {
  // Do any post-matmul cleanup work here.

  // Download C matrix from GPU
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}
