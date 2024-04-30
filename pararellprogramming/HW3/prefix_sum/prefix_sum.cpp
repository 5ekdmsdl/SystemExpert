#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <iostream>
#include <vector>
using namespace std;

void prefix_sum_sequential(double *out, const double *in, int N) {
  out[0] = in[0];
  for (int i = 1; i < N; ++i) {
    out[i] = in[i] + out[i - 1];
  }
}

struct argu{
  int pid;
  double* out;
  int N;
  int cnt;

  argu(int pid, double* out, int N, int cnt):pid(pid), out(out), N(N), cnt(cnt){};
};

void* runThread(void* ptr) {
  argu* arguments = (argu*)ptr;
  int iStart = arguments->N - 1;
  int iEnd = arguments->cnt;

  // std::cout << cnt << " : " << iEnd << " ~ " << iStart << std::endl;
  for (int i = iStart - arguments->pid; i >= iEnd; i -= arguments->cnt) {
    // std::cout << pid << " : " << i  << " += " << i-cnt << std::endl;
    arguments->out[i] += arguments->out[i - arguments->cnt];
  }

  return nullptr;
}


void prefix_sum_parallel(double *out, const double *in, int N) { 
    for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }
  
  vector<pthread_t*> threadP;
  for (int cnt = 1; cnt < N; cnt *= 2) {
    for (int pid = 0; pid < cnt; pid++) {
      threadP.push_back(new pthread_t());
      argu argument(pid, out, N, cnt);
      pthread_create(threadP[pid], nullptr, runThread, static_cast<void*>(&argument));
      // runThread(pid, out, N, cnt);
    }
    for(int pid = 0; pid < cnt; pid++){
      pthread_join(*threadP[pid], nullptr);
      delete threadP[pid];
    }
    threadP.clear();
  }
}
