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

static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

static void mat_mul_omp() {
  const int iTile = 64, jTile = 256, kTile = 256;

  if (M % iTile == 0 && N % jTile == 0 && K % kTile == 0) { 
    // printf("\n[LOG : rank %d] function start Barrier arrived \n", mpi_rank);
    // fflush(stdout);
    // CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD)); 

    if(mpi_rank == 0){
      #pragma omp parallel for num_threads(num_threads)
      for (int i = 0; i < M / 2; i += iTile) {
        for (int j = 0; j < N; j += jTile) {
          for (int k = 0; k < K; k += kTile) {
          
            for(int kk = k; kk < k + kTile; kk+=2){
              for(int ii = i; ii < i + iTile; ii++){
                  __m256 a0 = _mm256_set1_ps(A[(ii+0)*K+(kk+0)]);
                  __m256 a1 = _mm256_set1_ps(A[(ii+0)*K+(kk+1)]);
                for(int jj = j; jj < j + jTile; jj += 8){
                    __m256 c0 = _mm256_load_ps(&C[(ii+0) * N + jj]);

                    __m256 b0 = _mm256_load_ps(&B[(kk+0) * N + jj]);
                    __m256 b1 = _mm256_load_ps(&B[(kk+1) * N + jj]);
                
                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c0 = _mm256_fmadd_ps(a1, b1, c0);

                    _mm256_store_ps(&C[(ii+0)*N+jj], c0);
                  }
                }
              }
            }
          }
        }
    }
    else{
      #pragma omp parallel for num_threads(num_threads)
      for (int i = M / 2; i < M; i += iTile) {
        for (int j = 0; j < N; j += jTile) {
          for (int k = 0; k < K; k += kTile) {
          
            for(int kk = k; kk < k + kTile && kk < K; kk+=2){
              for(int ii = i; ii < i + iTile && ii < M; ii++){
                  __m256 a0 = _mm256_set1_ps(A[(ii+0)*K+(kk+0)]);
                  __m256 a1 = _mm256_set1_ps(A[(ii+0)*K+(kk+1)]);

                for(int jj = j; jj < j + jTile && jj < N; jj += 8){
                    __m256 c0 = _mm256_load_ps(&C[(ii+0) * N + jj]);

                    __m256 b0 = _mm256_load_ps(&B[(kk+0) * N + jj]);
                    __m256 b1 = _mm256_load_ps(&B[(kk+1) * N + jj]);
                
                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c0 = _mm256_fmadd_ps(a1, b1, c0);

                    _mm256_store_ps(&C[(ii+0) * N + jj], c0);
                  }
                }
              }
            }
          }
        }
    }
    // printf("rank %d ended its job. waiting ... \n", mpi_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    // if(mpi_rank == 0){
    //   printf("Start copying ... \n");
    // }

    if(mpi_rank == 1){
      MPI_Ssend(&C[M / 2 * N], M / 2 * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    else{
      MPI_Recv(&C[M / 2 * N], M / 2 * N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // printf("rank %d ended send/recv job. waiting ... \n", mpi_rank);

    MPI_Barrier(MPI_COMM_WORLD);
  }
  else {
    if(mpi_rank == 0){
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
    MPI_Barrier(MPI_COMM_WORLD);

    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size) {
  
  MPI_Barrier(MPI_COMM_WORLD);
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;    
  MPI_Barrier(MPI_COMM_WORLD);

  if(mpi_rank == 0){
    A = _A, B = _B, C = _C;    
  }
  else{   
    A = (float *)aligned_alloc(32, sizeof(float) * M * K);
    B = (float *)aligned_alloc(32, sizeof(float) * K * N);
    C = (float *)aligned_alloc(32, sizeof(float) * M * N);
    if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(A, M * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(C, M * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  
  // if(mpi_rank == 1){
  //   for(int i = 0; i < 10; i++){
  //     printf("%d %f %f %f  \n ", i, A[i], B[i], C[i]);
  //     fflush(stdout);
  //     MPI_Barrier(MPI_COMM_WORLD);
  //   }
  //   printf("\n");
  // }
  // MPI_Barrier(MPI_COMM_WORLD); 

  // TODO: parallelize & optimize matrix multiplication on multi-node
  // You must allocate & initialize A, B, C for non-root processes

  // FIXME: for now, only root process runs the matrix multiplication.
  // if (mpi_rank == 0)
  mat_mul_omp();
}

