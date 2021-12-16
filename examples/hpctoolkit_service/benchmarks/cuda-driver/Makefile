EXEC := main

CUBIN := vecSet1

INTERCEPTOR := intercept_kernel
INTERCEPT_FLAGS += -shared -Xcompiler -fPIC


CXX := /usr/local/cuda-11.1/bin/nvcc

ARCH := sm_70

CPU_FLAGS += -g -fopenmp $(CFLAGS)

GPU_FLAGS += -O3 -lineinfo -arch $(ARCH) $(INCLUDECUDA)

CUBIN_FLAGS += -cubin

LDFLAGS += -lcudart -lcuda -lstdc++ -lm $(HEADERNVMLAPI)

CXXFLAGS := -Xcompiler "$(CPU_FLAGS)" $(GPU_FLAGS)



ifneq ($(CUBIN),)
CUBIN_CUBIN := $(addsuffix .cubin, $(CUBIN))
endif



all: exec cubin interceptor

exec: $(EXEC)
$(EXEC): % : %.cu $(DEVICE_OBJ) $(INCLUDEPROG)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

cubin: $(CUBIN_CUBIN)
$(CUBIN_CUBIN): %.cubin : %.cu
	$(CXX) $(GPU_FLAGS) $(CUBIN_FLAGS) $<

interceptor:
# load and build hpctoolkit

run: all
	./$(EXEC)
	hpcrun -e gpu=nvidia -e REALTIME -t -o m ./$(EXEC)
	find ./m -not -name "$(EXEC)-*" -name "*-*" -delete
	hpcstruct $(EXEC)
	hpcstruct -j 32 ./m
	hpcprof2 -o db -S $(EXEC).hpcstruct  ./m

run-pc: all
	./$(EXEC)
	hpcrun -e gpu=nvidia,pc -o m ./$(EXEC)
	find ./m -not -name "$(EXEC)-*" -name "*-*" -delete
	hpcstruct $(EXEC)
	hpcstruct -j 32 --gpucfg no ./m
	hpcprof2 -o db -S $(EXEC).hpcstruct  ./m

clean:
	rm -rf $(EXEC) $(EXEC).hpcstruct *.o *.so *.dot *.hpcstruct *.cubin  m db*
