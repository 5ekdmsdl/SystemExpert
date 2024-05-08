#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define MAX_NUM_GPU 4
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  // if (i >= M || j >= N) return;

  float sum = 0;
  for(int k = 0; k < K; k++){
    sum += A[i * K + k] * B[k * N + j];  
  }
  C[i * N + j] = sum;
}


// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

void matmul(float *A, float *B, float *C, int M, int N, int K) {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  
  printf("Start mat mul ... \n");

  int blkSz = 2; int blkCnt = M * N / (blkSz * blkSz);

  int targetBlkSz = 8;
  if(M % targetBlkSz == 0 && N % targetBlkSz == 0 && K % targetBlkSz == 0){
    printf("optimized multiplication start ... \n"); fflush(stdout);
    blkSz = targetBlkSz;
    blkCnt = M * N / (blkSz * blkSz);  // = 8192 * 8192 / (64 * 64) = (128 * 128)

    if(blkCnt < SMCNT){
      printf("block count error !! \n"); fflush(stdout);
    }
    if((blkSz * blkSz) % WRAPSZ != 0){
      printf("block size error !! \n"); fflush(stdout);
    }  
  }

  printf("block size is %d * %d = %d \n", blkSz, blkSz, blkSz * blkSz); fflush(stdout);
  printf("grid size is %d * %d \n", M / blkSz, N / blkSz); fflush(stdout);

  dim3 blockDim(blkSz, blkSz, 1);
  dim3 gridDim(M / blkSz, N / blkSz, 1);

  printf("Start sgemm \n"); fflush(stdout);
  printf("M, N, K %d %d %d \n", M, N, K);
  matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  printf("Done sgemm \n"); fflush(stdout);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  printf("Sync Start ... \n"); fflush(stdout);
  CUDA_CALL(cudaDeviceSynchronize());
  printf("Sync Done \n"); fflush(stdout);
}

void matmul_initialize(int M, int N, int K) {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

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

void matmul_finalize() {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  
  printf("mat mul final ... \n");  fflush(stdout);
  // Download C matrix from GPU
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
  printf("mat mul final done ! \n");     fflush(stdout);
}
