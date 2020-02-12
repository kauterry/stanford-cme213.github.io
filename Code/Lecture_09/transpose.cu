#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <vector>
#include "utils.h"

const unsigned warp_size = 32;

__global__
void simpleTranspose(int* array_in, int* array_out, int n_rows, int n_cols) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;

  int col = tid % n_cols;
  int row = tid / n_cols;

  if(col < n_cols && row < n_rows) {
    array_out[col * n_rows + row] = array_in[row * n_cols + col];
  }
}

__global__
void simpleTranspose2D(int* array_in, int* array_out, int n_rows, int n_cols) {
  const int col = threadIdx.x + blockDim.x * blockIdx.x;
  const int row = threadIdx.y + blockDim.y * blockIdx.y;

  if(col < n_cols && row < n_rows) {
    array_out[col * n_rows + row] = array_in[row * n_cols + col];
  }
}

template<int num_warps>
__global__
void fastTranspose(int* array_in, int* array_out, int n_rows, int n_cols) {
  const int warp_id  = threadIdx.y;
  const int lane     = threadIdx.x;

  __shared__ int block[warp_size][warp_size+1];

  const int bc = blockIdx.x;
  const int br = blockIdx.y;

  // Load 32x32 block into shared memory
  int gc = bc * warp_size + lane; // Global column index

  for(int i = 0; i < warp_size / num_warps; ++i) {
    int gr = br * warp_size + i * num_warps + warp_id; // Global row index
    block[i * num_warps + warp_id][lane] = array_in[gr * n_cols + gc];
  }

  __syncthreads();

  // Now we switch to each warp outputting a row, which will read
  // from a column in the shared memory. This way everything remains
  // coalesced.
  int gr = br * warp_size + lane;

  for(int i = 0; i < warp_size / num_warps; ++i) {
    int gc = bc * warp_size + i * num_warps + warp_id;
    array_out[gc * n_rows + gr] = block[lane][i * num_warps + warp_id];
  }
}

void isTranspose(const std::vector<int>& A,
                 const std::vector<int>& B,
                 int n) {
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < n; ++j) {
      assert(A[n * i + j] == B[n * j + i]);
    }
  }
}

/* Must be an odd number */
#define MEMCOPY_ITERATIONS 11

int main(void) {
  const int n = (1<<12);

  printf("Number of MB to transpose: %ld\n\n",sizeof(int) * n * n / 1024 / 1024);

  int num_threads, num_blocks;

  std::vector<int> h_in(n * n);
  std::vector<int> h_out(n * n);

  for(int i = 0; i < n * n; ++i) {
    h_in[i] = random() % 100;
  }

  int* d_in, *d_out;
  checkCudaErrors(cudaMalloc(&d_in,  sizeof(int) * n * n));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * n * n));

  GpuTimer timer;
  timer.start();
  for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
    checkCudaErrors(cudaMemcpy(d_out, d_in, sizeof(int) * n * n,
			       cudaMemcpyDeviceToDevice));
  }
  timer.stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  printf("Bandwidth bench\n");
  printf("GPU took %g ms\n",timer.elapsed() / MEMCOPY_ITERATIONS);
  printf("Effective bandwidth is %g GB/s\n",
	 (2*sizeof(int)*n*n*MEMCOPY_ITERATIONS)/(1e9*1e-3*timer.elapsed()));

  checkCudaErrors(cudaMemcpy(d_in, &h_in[0], sizeof(int) * n * n,
			     cudaMemcpyHostToDevice));

  num_threads = 256;
  num_blocks = (n * n + num_threads - 1) / num_threads;

  timer.start();
  for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
    simpleTranspose<<<num_blocks, num_threads>>>(d_in, d_out, n, n);
  }
  timer.stop();

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&h_out[0], d_out, sizeof(int) * n * n,
			     cudaMemcpyDeviceToHost));

  isTranspose(h_in, h_out, n);

  printf("\nsimpleTranspose\n");
  printf("GPU took %g ms\n",timer.elapsed());
  printf("Effective bandwidth is %g GB/s\n",
	 (2*sizeof(int)*n*n*MEMCOPY_ITERATIONS)/(1e9*1e-3*timer.elapsed()));

  dim3 block_dim(8, 32);
  dim3 grid_dim(n / 8, n / 32);

  timer.start();
  for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
    simpleTranspose2D<<<grid_dim, block_dim>>>(d_in, d_out, n, n);
  }
  timer.stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&h_out[0], d_out, sizeof(int) * n * n,
			     cudaMemcpyDeviceToHost));

  isTranspose(h_in, h_out, n);

  printf("\nsimpleTranspose2D\n");
  printf("GPU took %g ms\n",timer.elapsed());
  printf("Effective bandwidth is %g GB/s\n",
	 (2*sizeof(int)*n*n*MEMCOPY_ITERATIONS)/(1e9*1e-3*timer.elapsed()));

  const int num_warps_per_block = 256/32;
  assert(warp_size % num_warps_per_block == 0);
  block_dim.x = warp_size;
  block_dim.y = num_warps_per_block;
  grid_dim.x = n / warp_size;
  grid_dim.y = n / warp_size;

  timer.start();
  for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
    fastTranspose<num_warps_per_block><<<grid_dim, block_dim>>>(d_in, d_out, n,
								n);
  }
  timer.stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&h_out[0], d_out, sizeof(int) * n * n,
			     cudaMemcpyDeviceToHost));

  isTranspose(h_in, h_out, n);

  printf("\nfastTranspose\n");
  printf("GPU took %g ms\n",timer.elapsed());
  printf("Effective bandwidth is %g GB/s\n",
	 (2*sizeof(int)*n*n*MEMCOPY_ITERATIONS)/(1e9*1e-3*timer.elapsed()));

  checkCudaErrors(cudaFree(d_in));
  checkCudaErrors(cudaFree(d_out));

  return 0;
}
