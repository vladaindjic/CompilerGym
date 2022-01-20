// #include "header.h"
// #include <stdbool.h>
// #include <stdio.h>
#include <sys/time.h>

#define BENCH(NAME, RUN_LINE, ITER, DIGEST_LINE) \
  {                                              \
    struct timeval Start, End;                   \
    RUN_LINE;                                    \
    gettimeofday(&Start, 0);                     \
    for (int i = 0; i < (ITER); ++i) RUN_LINE;   \
    gettimeofday(&End, 0);                       \
    unsigned r = DIGEST_LINE;                    \
    long mtime, s, us;                           \
    s = End.tv_sec - Start.tv_sec;               \
    us = End.tv_usec - Start.tv_usec;            \
    mtime = (s * 1000 + us / 1000.0) + 0.5;      \
    printf("%ld", mtime);                        \
  }

#ifndef N
#define N 1000000
#endif

#ifndef n
#define n 3
#endif

int A[N];

__attribute__((noinline)) void example1(int* ret) {
  //#pragma unroll(n)
  for (int i = 0; i < N - 3; i++) A[i] = A[i + 1] + A[i + 2] + A[i + 3];

  *ret = A[N - 1];
}

__attribute__((optnone)) int main(int argc, char* argv[]) {
  int dummy = 0;
  // TODO: initialize tensors
  BENCH("example1", example1(&dummy), 100, dummy);

  return 0;
}
