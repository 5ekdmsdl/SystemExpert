#include "mat_mul.h"

#include <omp.h>
#include <immintrin.h>
#include <stdio.h>

#include <sched.h>

static float *A, *B, *C;
static int M, N, K;
static int num_threads;

#define ITILESIZE (64)
#define JTILESIZE (256)
#define KTILESIZE (256)

static void mat_mul_omp() {
if (M % ITILESIZE == 0 && N % JTILESIZE == 0 && K % KTILESIZE == 0) {
  #pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M; i += ITILESIZE) {
      for (int j = 0; j < N; j += JTILESIZE) {
        for (int k = 0; k < K; k += KTILESIZE) {

          for (int kk = k; kk < k + KTILESIZE; kk+=2) {
            for (int ii = i; ii < i + ITILESIZE; ii++) {
              __m256 a0 = _mm256_set1_ps(A[(ii+0)*K+(kk+0)]);
              __m256 a1 = _mm256_set1_ps(A[(ii+0)*K+(kk+1)]);
            
              for (int jj = j; jj < j + JTILESIZE; jj+=8) {
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
      for (int ii = 0; ii < M; ii += ITILESIZE) {
        for (int kk = 0; kk < K; kk += KTILESIZE) {
          for (int jj = 0; jj < N; jj += JTILESIZE) {
            for (int i = ii; i < ii + ITILESIZE && i < M; ++i) {
              for (int k = kk; k < kk + KTILESIZE && k < K; ++k) {
                for (int j = jj; j < jj + JTILESIZE && j < N; ++j) {
                  C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
              }
            }
          }
        }
      }
    }
  
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads) {
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads;

  mat_mul_omp();
}
