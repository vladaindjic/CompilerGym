
// TODO: use templates instead of macros
#ifndef N
#define N 32
#endif

#ifndef Ih
#define Ih 3
#endif

#ifndef Iw
#define Iw 12
#endif

#ifndef Ic
#define Ic 12
#endif

#ifndef Oc
#define Oc 64
#endif

#ifndef Kh
#define Kh 3
#endif

#ifndef Kw
#define Kw 3
#endif

// TODO: include pad, stride, and dilation
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>

/**
 * Warmup and then measure.
 *
 * Adapted from Neurovectorizer's implementation:
 * https://github.com/intel/neuro-vectorizer/blob/d1b068998c08865c59f1586845bb947229f70a51/training_data/header.h
 *
 * Which was in turn adapted from LLVM:
 * https://github.com/llvm/llvm-test-suite/blob/7eca159e29ca4308256ef6e35560a2d884ac6b01/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp#L330-L336
 */
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

#define Oh Ih - Kh + 1
#define Ow Iw - Kw + 1

float x[N][Ih][Iw][Ic];
float w[Oc][Kh][Kw][Ic];
float y[N][Oh][Ow][Oc];

__attribute__((noinline))
// template <N=32, Iw=...>#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>

/**
 * Warmup and then measure.
 *
 * Adapted from Neurovectorizer's implementation:
 * https://github.com/intel/neuro-vectorizer/blob/d1b068998c08865c59f1586845bb947229f70a51/training_data/header.h
 *
 * Which was in turn adapted from LLVM:
 * https://github.com/llvm/llvm-test-suite/blob/7eca159e29ca4308256ef6e35560a2d884ac6b01/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp#L330-L336
 */
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
void conv2d(int* ret) {
  // loop over output
  for (int n = 0; n < N; n++) {
    for (int oh = 0; oh < Oh; oh++) {
      for (int ow = 0; ow < Ow; ow++) {
        for (int oc = 0; oc < Oc; oc++) {
          y[n][oh][ow][oc] = 0;
// loop over filter
#pragma unroll(Kh)
          for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
              for (int ic = 0; ic < Iw; ic++) {
                // TODO: include pad, stride, and dilation
                y[n][oh][ow][oc] += w[oc][kh][kw][ic] * x[n][oh - kh + 1][ow - kw + 1][ic];
              }
            }
          }
        }
      }
    }
  }
  *ret = y[N - 1][Oh - 1][Ow - 1][Oc - 1];
}

__attribute__((optnone)) int main(int argc, char* argv[]) {
  int dummy = 0;
  // TODO: initialize tensors
  BENCH("conv2d", conv2d(&dummy), 100, dummy);

  return 0;
}
