#include "iostream"
#define SZ 16

void func(double *out, double *in, int N) {
  for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }

  int cnt = 1;
  while (cnt < N) {
    for (int i = N - 1; i >= cnt; i--) {
      out[i] += out[i - cnt];
    }
    cnt *= 2;
  }
}

int main() {
  double out[SZ] = {0};
  double in[SZ] = {0};
  for (int i = 0; i < SZ; i++) {
    in[i] = rand() % 100;
    // std::cout << in[i] << " ";
  }
  // std::cout << std::endl;
  func(out, in, SZ);

  int result = 1;
  int sum = 0;
  for (int i = 0; i < SZ; i++) {
    sum += in[i];
    if (sum != out[i])
      result = 0;
    // std::cout << i << " : " << in[i] << " " << sum << " " << out[i]  << " "
    // << (sum != out[i]) << std::endl;
  }
  std::cout << "result : " << result << std::endl;
}