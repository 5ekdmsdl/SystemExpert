#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>

#define MAX_NUM_THREADS (4)


void prefix_sum_sequential(double *out, const double *in, int N) {
  out[0] = in[0];
  for (int i = 1; i < N; ++i) {
    out[i] = in[i] + out[i - 1];
  }
}


static pthread_barrier_t barrier;
static long NUM_THREADS;
static double *out;
static const double *in;
static int N;

void* prefix_sum_parallel_thread(void* i) {
	long tid = (long) i;

	for (long id = tid; id < N; id += NUM_THREADS) {
		out[id] = in[id];
	}
	pthread_barrier_wait(&barrier);

	long st = 1;
	while (st < N) {
		for (long id = N - 1 - tid; id >= 0; id -= NUM_THREADS) {
			float val = 0.;
			if (id >= st) {
				val = out[id] + out[id-st];
			}
			pthread_barrier_wait(&barrier);

			if (id >= st) {
				out[id] = val;
			}
			pthread_barrier_wait(&barrier);
		}
		st *= 2;
	}
	return NULL;
}

void prefix_sum_parallel(double *_out, const double *_in, int _N) {
	out = _out;
	in = _in;
	N = _N;
	NUM_THREADS = (N < MAX_NUM_THREADS) ? N : MAX_NUM_THREADS;
	pthread_t threads[NUM_THREADS];
  pthread_attr_t attr[NUM_THREADS];
  cpu_set_t cpus[NUM_THREADS];
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);

	for (long i = 0; i < NUM_THREADS; ++i) {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
		pthread_create(&threads[i], NULL, prefix_sum_parallel_thread, (void*)i);
	}
	for (long i = 0; i < NUM_THREADS; ++i) {
		pthread_join(threads[i], NULL);
	}
	pthread_barrier_destroy(&barrier);
}
