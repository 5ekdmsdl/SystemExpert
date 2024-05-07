#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

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

  // int kTile = 128;
  C[i * N + j] = 0;
  for(int k = 0; k < K; k++){
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}

// Device (GPU) pointers
static float *a_d;
static float *b_d;
static float *c_d;

void mat_mul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Launch kernel on every GPU
  printf("Start mat mul ... \n");
  int count;
  cudaGetDeviceCount(&count);
  printf("Number of devices: %d\n", count);

  printf("Getting dev info ... \n");
  cudaDeviceProp props[10];
  for (int i = 0; i < count; ++i) {
    printf("\tdevice %d:\n", i);
    cudaGetDeviceProperties(&props[i], i);
    printf("\t\tname: %s\n", props[i].name);
    printf("\t\tmultiProcessorCount: %d\n", props[i].multiProcessorCount);
    printf("\t\tmaxThreadsPerBlock: %d\n", props[i].maxThreadsPerBlock);
    printf("\t\ttotalGlobalMem: %lu\n", props[i].totalGlobalMem);
    printf("\t\tsharedMemPerBlock: %lu\n", props[i].sharedMemPerBlock);
  }

  int targetBlkSz = 128;
  int blkSz = 1;
  if(M % targetBlkSz == 0 && N % targetBlkSz == 0 && K % targetBlkSz == 0){
    printf("optimized multiplication start ... \n"); fflush(stdout);
    blkSz = targetBlkSz;
    int blkCnt = M * N / (blkSz * blkSz);  // = 8192 * 8192 / (64 * 64) = (128 * 128)
    printf("block size is %d \n", blkSz * blkSz); fflush(stdout);
    printf("the number of block is %d \n", blkCnt); fflush(stdout);

    if(blkCnt < props[0].multiProcessorCount){
      printf("block count error !! \n"); fflush(stdout);
    }
    if(blkSz % 32 != 0){
      printf("block size error !! \n"); fflush(stdout);
    }
  }
  
  dim3 blockDim(blkSz, 1, 1);
  dim3 gridDim(M, N, 1);

  printf("Start sgemm \n"); fflush(stdout);
  printf("M, N, K %d %d %d \n", M, N, K);
  sgemm<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  printf("Done sgemm \n"); fflush(stdout);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  // printf("Sync Start ... \n"); fflush(stdout);
  // CUDA_CALL(cudaDeviceSynchronize());
  // printf("Sync Done \n"); fflush(stdout);
}

void mat_mul_init(float *A, float *B, float *C, int M, int N, int K) {
  // Allocate device memory
  // M 8196, N 8196, K 8196
  printf("mat mul init ... \n");    
  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));

  // Upload A and B matrix to GPU
  CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
  printf("mat mul init done ! \n");   fflush(stdout);  
}

void mat_mul_final(float *A, float *B, float *C, int M, int N, int K) {
  // Do any post-matmul cleanup work here.

  printf("mat mul final ... \n");  fflush(stdout);
  // Download C matrix from GPU
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
  printf("mat mul final done ! \n");     fflush(stdout);
}
