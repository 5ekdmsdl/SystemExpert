#include <cuda_runtime.h>

#include <cstdio>

#include "mat_mul.h"

#define BLOCK 32

#define CUDA_CALL(f)                                                       \
  {                                                                        \
    cudaError_t err = (f);                                                 \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__, \
              err, cudaGetErrorString(err));                               \
      exit(1);                                                             \
    }                                                                      \
  }

// Device (GPU) pointers
static float *a_d;
static float *b_d;
static float *c_d;

// sgemm kernel0
__global__ void sgemm0(float *A, float *B, float *C, int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x; 
  int by = blockIdx.y;

  __shared__ float Ashared[BLOCK][BLOCK];
  __shared__ float Bshared[BLOCK][BLOCK];

  int Row = by * BLOCK + ty; // blockDim.y * blockIdx.y + threadIdx.y;
  int Col = bx * BLOCK + tx; // blockDim.x * blockIdx.x + threadIdx.x;

  float value = 0;

  // In the each loop, threads in the block load just one value in the global memory (A, B) to the shared memory (Ashread, Bshared)
  // Check how convert the tile indx to the global index
  // tk * BLOCK : absolute position when tile is moved by tk loop
  // ty, tx : relative position in the tile, that is same as thread id in the block (threadIdx.x, threadIdx.y)
  //// ty : vertical position
  //// tx : horizontal position
  for (int tk=0; tk < (K+BLOCK-1) / BLOCK ; ++tk) {

    if ((Row < M) && ((tk * BLOCK + tx) < K)) Ashared[ty][tx] = A[Row * K + (tk * BLOCK + tx)];
    else Ashared[ty][tx] = 0;

    if ((Col < N) && ((tk * BLOCK + ty) < K)) Bshared[ty][tx] = B[(tk * BLOCK + ty)*N + Col];
    else Bshared[ty][tx] = 0;

    __syncthreads();

    for (int k = 0; k < BLOCK; ++k) {
        value += Ashared[ty][k] * Bshared[k][tx];
    }
    __syncthreads();
  }

  if ((Row < M) && (Col < N))
    C[Row*N + Col] = value;
}


// sgemm kernel1
__global__ void sgemm1(float *A, float *B, float *C, int M, int N, int K) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;

  float acc = 0;

  for (int k = 0; k < K; ++k) {
    acc += A[i * K + k] * B[k * N + j];
  }

  C[i * N + j] = acc;
}

// sgemm kernel2
__global__ void sgemm2(float *A, float *B, float *C, int M, int N, int K) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;

  C[i * N + j] = 0;
  for (int k = 0; k < K; ++k) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}

// sgemm kernel3
__global__ void sgemm3(float *A, float *B, float *C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;

  C[i * N + j] = 0;
  for (int k = 0; k < K; ++k) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}

// Skeleton + thread block size optimization + Memory Coalescing + Register Re-use + Shared Memory
void mat_mul0(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  dim3 blockDim(BLOCK, BLOCK, 1);
  dim3 gridDim((N + BLOCK - 1) / BLOCK, (M + BLOCK-1) / BLOCK, 1);

  sgemm0<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL( cudaDeviceSynchronize() );
}

// Skeleton + thread block size optimization + Memory Coalescing + Register Re-use
void mat_mul1(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  dim3 blockDim(BLOCK, BLOCK, 1);
  dim3 gridDim((N + BLOCK -1) / BLOCK , (M + BLOCK - 1) / BLOCK, 1);

  sgemm1<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

// Skeleton + thread block size optimization + Memory Coalescing
void mat_mul2(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  dim3 blockDim(BLOCK, BLOCK, 1);
  dim3 gridDim((N + BLOCK -1) / BLOCK , (M + BLOCK - 1) / BLOCK, 1);

  sgemm2<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

// Skeleton + thread block size optimization
void mat_mul3(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  dim3 blockDim(BLOCK, BLOCK, 1);
  dim3 gridDim((M + BLOCK -1) / BLOCK , (N + BLOCK - 1) / BLOCK, 1);

  sgemm3<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

// Skeleton
void mat_mul4(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  dim3 blockDim(1, 1, 1);
  dim3 gridDim(M, N, 1);

  sgemm3<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

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
