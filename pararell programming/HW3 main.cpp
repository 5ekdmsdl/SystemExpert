#include "iostream"
#include <pthread.h>
#include <vector>
#define SZ 16
#define nthread 4
using namespace std;

struct argu{
  int pid;
  double* out;
  int N;
  int cnt;

  argu(int pid, double* out, int N, int cnt):pid(pid), out(out), N(N), cnt(cnt){};
};

void* runThread(void* ptr) {
  argu* arguments = (argu*)ptr;
  int part = arguments->N / nthread;
  int iStart = (nthread - arguments->pid) * part - 1;
  int iEnd = std::max(arguments->cnt, (nthread - (arguments->pid + 1)) * part);
  if(iStart < iEnd) return nullptr;

  std::cout << arguments->pid << " : " << iEnd << " ~ " << iStart << std::endl;
  for (int i = iStart; i >= iEnd; i--) {
    arguments->out[i] += arguments->out[i - arguments->cnt];
  }

  return nullptr;
}

void func(double *out, double *in, int N) {
  for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }

  vector<pthread_t*> threadP;
  for (int cnt = 1; cnt < N; cnt *= 2) {
    for (int pid = 0; pid < nthread; pid++) {
      threadP.push_back(new pthread_t());
      argu* argument = new argu(pid, out, N, cnt);
      pthread_create(threadP[pid], nullptr, runThread, static_cast<void*>(argument));
    }

    for(int pid = 0; pid < nthread; pid++){
      pthread_join(*threadP[pid], nullptr);
    }
    std::cout << cnt << " thread " << " joined" << std::endl;

    threadP.clear();
  }


}

int main() {
  double out[SZ] = {0};
  double in[SZ] = {0};
  for (int i = 0; i < SZ; i++) {
    in[i] = rand() % 100;
  }
  func(out, in, SZ);

  int result = 1;
  int sum = 0;
  for (int i = 0; i < SZ; i++) {
    sum += in[i];
    if (sum != out[i])
      result = 0;
  }
  std::cout << "result : " << result << std::endl;
}