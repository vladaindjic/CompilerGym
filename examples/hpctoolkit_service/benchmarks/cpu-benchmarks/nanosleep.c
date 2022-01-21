// #include <stdbool.h>
#include <time.h>
#include "header.h"

#define ITER 10

__attribute__((noinline))
void sleep_func(int* ret) {
  struct timespec ts = {0, 0.1};

  for (int ic = 0; ic < ITER; ic++) { 
    while (clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, &ts) != 0);
  }

}

__attribute__((optnone)) int main(int argc, char* argv[]) {
  int dummy = 0;
  // TODO: initialize tensors
  BENCH("nanosleep", sleep_func(&dummy), 100, dummy);

  return 0;
}
