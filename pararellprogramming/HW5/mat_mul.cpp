#include "mat_mul.h"
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <sched.h>
#include "mat_mul.h"
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include "mpi_error_handler.h"

#define VECTORSIZE 8
static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

static void mat_mul_omp() {
  //FIXME: Optimize the following code using OpenMP

  int signal = 1;
  const int iTile = 64, jTile = 256, kTile = 256;
  int i = 0, j = 0, k = 0;
  if (M % iTile == 0 && N % jTile == 0 && K % kTile == 0) { 

  if(mpi_rank == 0){
    printf("Rank 0 Start @@@@@@@@@@@@@@@@@@@@@ \n\n"); 
    #pragma omp parallel for num_threads(num_threads)
    for (i = 0; i < M; i += iTile) {
      for (j = 0; j < N; j += jTile) {
        for (k = 0; k < K; k += kTile) {
         
          for(int kk = k; kk < k + kTile; kk+=2){
            for(int ii = i; ii < i + iTile; ii++){
                __m256 a0 = _mm256_set1_ps(A[(ii+0)*K+(kk+0)]);

                for(int jj = j; jj < j + jTile; jj += 8){
                  __m256 b0 = _mm256_load_ps(&B[(kk+0) * N + jj]);
                  __m256 c0 = _mm256_load_ps(&C[(ii+0) * N + jj]);
                  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
                  printf("Ready to MPI_Ssend bbbbbbbbbbbbbbbbbbbbbbbbbbb \n");
                  CHECK_MPI(MPI_Ssend(&signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD));
                  printf("send bbbbbbbbbbbbbbbbbbbbbbbbbbb \n");
                  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
              
                  c0 = _mm256_fmadd_ps(a0, b0, c0);
                  // _mm256_store_ps(&C[(ii+0)*N+jj], c0);
                } 
            }
          }
        }
      }
    }
  }
  else{
    printf("Rank 1 Start @@@@@@@@@@@@@@@@@@@@@ \n\n"); 
    #pragma omp parallel for num_threads(num_threads)
    for (i = 0; i < M; i += iTile) {
      for (j = 0; j < N; j += jTile) {
        for (k = 0; k < K; k += kTile) {

          for(int kk = k; kk < k + kTile; kk+=2){
            for(int ii = i; ii < i + iTile; ii++){
              __m256 a1 = _mm256_set1_ps(A[(ii+0)*K+(kk+1)]);

              for(int jj = j; jj < j + jTile; jj += 8){
                  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
                  printf("Ready to MPI_Recv bbbbbbbbbbbbbbbbbbbbbbbbbbb \n");
                  CHECK_MPI(MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                  printf("MPI_Recv bbbbbbbbbbbbbbbbbbbbbbbbbbb \n");
                  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

                __m256 b1 = _mm256_load_ps(&B[(kk+1) * N + jj]);
                __m256 c0 = _mm256_load_ps(&C[(ii+0) * N + jj]);

                c0 = _mm256_fmadd_ps(a1, b1, c0);
                printf("%d : %f \n", omp_get_thread_num(), c0);

                float* sendBuf = (float *)aligned_alloc(32, sizeof(float) * (VECTORSIZE + 10));
                if (sendBuf == NULL) {
                  printf("Failed to allocate memory for matrix.\n");
                  exit(0);
                }
                

                // printf("[ ");
                // for (int i = 0; i < 8; i++) {
                //     printf("%f ", sendBuf[i]);
                // }
                // printf("]\n");
                
                // _mm256_storeu_ps(sendBuf, c0);
                // _mm256_store_ps(sendBuf, c0);
                // MPI_Send(sendBuf, VECTORSIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
              }
            }
          }
        }
      }
    }
  }
  }
  else {
    #pragma omp parallel for num_threads(num_threads)
      for (int ii = 0; ii < M; ii += iTile) {
        for (int kk = 0; kk < K; kk += kTile) {
          for (int jj = 0; jj < N; jj += jTile) {
            for (int i = ii; i < ii + iTile && i < M; ++i) {
              for (int k = kk; k < kk + kTile && k < K; ++k) {
                for (int j = jj; j < jj + jTile && j < N; ++j) {
                  C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
              }
            }
          }
        }
      }
    }
    printf("rank %d barrier bbbbbbbbbb\n", mpi_rank);
    MPI_Barrier(MPI_COMM_WORLD);
  }

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size) {
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  // TODO: parallelize & optimize matrix multiplication on multi-node
  // You must allocate & initialize A, B, C for non-root processes

  // FIXME: for now, only root process runs the matrix multiplication.
  // if (mpi_rank == 0)
  mat_mul_omp();
}

