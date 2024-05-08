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
  // FILL IN HERE
}


// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

int devCnt = 0;

void matmul(float *A, float *B, float *C, int M, int N, int K) {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  
  if(mpi_rank == 0){
    for (int i = 0; i < devCnt; i++) {
      cudaSetDevice(i);
      cudaMemcpy(a_d[i], A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
      cudaMemcpy(b_d[i], B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
      printf("dev %d copy done ! \n ", i); 
    }
  }
}

void matmul_initialize(int M, int N, int K) {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  if(mpi_rank == 0){
    cudaGetDeviceCount(&devCnt);

    for (int i = 0; i < devCnt; i++) {
      cudaSetDevice(i);
      // cudaStreamCreate(&stream[i]);
      // cudaEventCreate(&events[i]);

      cudaMalloc(&a_d[i], sizeof(float) * M * K);
      cudaMalloc(&b_d[i], sizeof(float) * K * N);
      cudaMalloc(&c_d[i], sizeof(float) * M * N);

      printf("dev %d malloc Done ! \n", i);
    }
  }
}

void matmul_finalize() {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  
  // FILL IN HERE
}
