#include "mat_mul.h"

#include <omp.h>
#include <immintrin.h>
#include <stdio.h>

#include <sched.h>

static float *A, *B, *C;
static int M, N, K;
static int num_threads;

static void mat_mul_omp() {
  //FIXME: Optimize the following code using OpenMP

  #pragma omp parallel for num_threads(num_threads)
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
	      // printf("%d %d %d \n", k, i,j);
        C[i * N + j] += A[i * K + k] * B[k * N + j];
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
