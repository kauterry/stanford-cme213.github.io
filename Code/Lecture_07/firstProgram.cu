#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include "utils.h"

using std::vector;

__device__ __host__
int f(int i) {
    return i*i;
}

__global__
void kernel(int* out) {
    out[threadIdx.x] = f(threadIdx.x);
}

int main(int argc, char** argv) {
    int N = 32;

    if(argc == 2) {
        N = atoi(argv[1]);
    }

    int* d_output;

    /* checkCudaErrors:
       A wrapper function we wrote to test whether an error occurred
       when launching a kernel.
       cudaMalloc:
       Allocated memory on device
       */
    checkCudaErrors(cudaMalloc(&d_output, sizeof(int) * N));

    /* This is like a parallel for loop.
       kernel is the function above.
       d_output is the input variable.
       This call will execute the function kernel using N threads.
       Each thread gets a different threadIdx.x value.
       */
    kernel<<<1, N>>>(d_output);

    /* This is just to check that the kernel executed as expected. */
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    vector<int> h_output(N);
    /* This function copies the data back from GPU to CPU.
       See cudaMemcpyDeviceToHost
       You also have
       cudaMemcpyHostToDevice
       */
    checkCudaErrors(cudaMemcpy(&h_output[0], d_output, sizeof(int) * N,
                               cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; ++i) {
        if (i==0 || i==N-1 || i%(N/10) == 0)
		printf("Entry %3d, written by thread %2d\n", h_output[i], i);
        assert(h_output[i] == f(i));
    }

    /* Free memory on the device. */
    checkCudaErrors(cudaFree(d_output));

    return 0;
}

#if 0
int* d_output;
cudaMalloc(&d_output, sizeof(int) * N);
kernel<<<1, N>>>(d_output);
vector<int> h_output(N);
cudaMemcpy(&h_output[0], d_output, sizeof(int) * N,
           cudaMemcpyDeviceToHost);
for(int i = 0; i < N; ++i) {
   printf("Entry %3d, written by thread %2d\n",
          h_output[i], i);
}
cudaFree(d_output);
#endif
