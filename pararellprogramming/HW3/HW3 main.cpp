#include "iostream"
#include <pthread.h>
#include <vector>

#define SZ 1048
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
  int iStart = arguments->N - 1;
  int iEnd = arguments->cnt;

  // std::cout << cnt << " : " << iEnd << " ~ " << iStart << std::endl;
  for (int i = iStart - arguments->pid; i >= iEnd; i -= arguments->cnt) {
    // std::cout << pid << " : " << i  << " += " << i-cnt << std::endl;
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
    for (int pid = 0; pid < cnt; pid++) {
      threadP.push_back(new pthread_t());
      argu* argument = new argu(pid, out, N, cnt);
      pthread_create(threadP[pid], nullptr, runThread, static_cast<void*>(argument));
      // runThread(pid, out, N, cnt);
    }
    for(int pid = 0; pid < cnt; pid++){
      pthread_join(*threadP[pid], nullptr);
    }
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