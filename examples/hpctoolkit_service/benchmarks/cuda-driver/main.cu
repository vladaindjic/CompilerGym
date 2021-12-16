#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../utils/common.h"


static size_t N = 10;


void init(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    p[i] = i;
  }
}


void output(int *p, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("index %zu: %d\n", i, p[i]);
  }
}


int main(int argc, char *argv[]) {

  // Init device
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  int device_id = 0;

  cu_init_device(device_id, device, context);
  // cu_load_module_function(module, "vecSet_1.cubin", function, "vecSet_1");
  cu_load_module_function(module, "vecSet1.cubin", function, "vecExpensive1");


  int l[N];
  CUdeviceptr dl;

  init(l, N);

  size_t threads = 256;
  size_t blocks = (N - 1) / threads + 1;

  DRIVER_API_CALL(cuCtxSetCurrent(context));

  DRIVER_API_CALL(cuMemAlloc(&dl, N * sizeof(int)));
  DRIVER_API_CALL(cuMemcpyHtoD(dl, l, N * sizeof(int)));

  void *args[4] = {
    &dl, &N
  };

  DRIVER_API_CALL(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));

  DRIVER_API_CALL(cuMemcpyDtoH(l, dl, N * sizeof(int)));

  DRIVER_API_CALL(cuMemFree(dl));


  output(l, N);

  DRIVER_API_CALL(cuCtxSynchronize());


  DRIVER_API_CALL(cuModuleUnload(module));
  DRIVER_API_CALL(cuCtxDestroy(context));
  RUNTIME_API_CALL(cudaDeviceSynchronize());

  return 0;
}
