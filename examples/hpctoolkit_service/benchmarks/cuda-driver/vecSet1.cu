extern "C"

__global__
void vecExpensive1(int *l, size_t N) {
  size_t cyc_limit = 10 * 1E6 * N;
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  clock_t start = clock();
  volatile clock_t now;
  for (;;) {
    now = clock();
    clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
    if (cycles >= cyc_limit) {
      break;
    }
  }

  if (idx < N) {
    l[idx] = 1;
  }
}
