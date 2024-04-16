#include "iostream"
#define SZ 16
#define nthread 4

void runThread(int pid, double *out, int N, int cnt) {
  int part = N / nthread;
  int iStart = (nthread - pid) * part - 1;
  int iEnd = std::max(cnt, (nthread - (pid + 1)) * part);

  // std::cout << cnt << " : " << iEnd << " ~ " << iStart << std::endl;
  for (int i = iStart; i >= iEnd; i--) {
    out[i] += out[i - cnt];
  }
}

void func(double *out, double *in, int N) {
  for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }

  for (int cnt = 1; cnt < N; cnt *= 2) {
    for (int pid = 0; pid < nthread; pid++) {
      runThread(pid, out, N, cnt);
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