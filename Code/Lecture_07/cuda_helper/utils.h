#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

struct GpuTimer
{
  cudaEvent_t start_;
  cudaEvent_t stop_;

  GpuTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start()
  {
    cudaEventRecord(start_, 0);
  }

  void stop()
  {
    cudaEventRecord(stop_, 0);
  }

  float elapsed()
  {
    float elapsed_;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_, start_, stop_);
    return elapsed_;
  }
};
#endif
