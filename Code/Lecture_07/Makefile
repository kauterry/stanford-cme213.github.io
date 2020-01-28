CXX=nvcc
CXXFLAGS=-Icuda_helper -O3 --compiler-options -Wall -arch=sm_35

SRC := $(wildcard *.cpp) $(wildcard *.cu)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC))) $(patsubst %.cu,%,$(filter %.cu,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

% : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(EXE)
