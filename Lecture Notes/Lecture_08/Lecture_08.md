class: center, middle

# CME 213, ME 339 - Winter 2020

## Eric Darve, ICME

![:width 40%](Stanford.jpg)

“If debugging is the process of removing bugs, then programming must be the process of putting them in.”
(Edsger W. Dijkstra)

---
class: center, middle

# Let's get started!

---
class: middle

1. Setup Google Cloud Platform
2. Run the script [$ ./create_vm_gpu1.sh](https://github.com/stanford-cme213/stanford-cme213.github.io/blob/master/Code/create_vm_gpu1.sh)
3. Log on the instance: `$ gcloud compute ssh gpu1`

---
class: middle

Copy the file [Lecture_08.zip](https://github.com/stanford-cme213/stanford-cme213.github.io/blob/master/Code/Lecture_08.zip) to `gpu1`

On your VM:

`$ unzip Lecture_08.zip`

`$ make`

---
class: middle

Run

`./deviceQuery`

`./bandwidthTest`

---
class: middle

```
darve@gpu1:~/Lecture_08$ ./deviceQuery 
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla K80"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    3.7
  Total amount of global memory:                 11441 MBytes (11996954624 bytes)
  (13) Multiprocessors, (192) CUDA Cores/MP:     2496 CUDA Cores
  GPU Max Clock rate:                            824 MHz (0.82 GHz)
  Memory Clock rate:                             2505 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 1572864 bytes
  ...
```

---
class: middle

```
darve@gpu1:~/Lecture_08$ ./bandwidthTest 
 [...]
 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   32000000			7.9

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   32000000			10.7

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(GB/s)
   32000000			157.1

Result = PASS
```

---
class: center, middle

[firstProgram.cu](https://github.com/stanford-cme213/stanford-cme213.github.io/blob/master/Code/Lecture_08/firstProgram.cu)

---
class: center, middle

`checkCudaErrors(...)`

CUDA functions often fail silently

Use this to check for errors before continuing

---
class: middle

```
    int* d_output;

    cudaMalloc(&d_output, sizeof(int) * N);

    kernel<<<1, N>>>(d_output);

    vector<int> h_output(N);
    cudaMemcpy(&h_output[0], d_output, sizeof(int) * N,
                               cudaMemcpyDeviceToHost);

    cudaFree(d_output);
```    

---
class: middle

```
kernel<<<1, N>>>(d_output);
```

`N` : number of threads to launch for function `kernel`

Threads are numbered 0 to $N-1$

---
class: middle

```
__device__ __host__
int f(int i) {
    return i*i;
}

__global__
void kernel(int* out) {
    out[threadIdx.x] = f(threadIdx.x);
}
```

---
class: center, middle

# `global` / `host` / `device`

 ???

---
class: middle

`__global__` kernel will be

- Executed on the device
- Callable from the host

---
class: middle

`__host__` kernel will be

- Executed on the host
- Callable from the host

---
class: middle

`__device__` kernel will be

- Executed on the device
- Callable from the device only

---
class: center, middle

Get information about the current thread

Use the built-in variable `threadIdx`

We will learn more about this later

---
class: middle

Run

```
darve@gpu1:~/Lecture_08$ ./firstProgram -N=32
Using 32 threads = 1 warps
Entry          0, written by thread     0
...
Entry        961, written by thread    31
```

---
class: middle

```
darve@gpu1:~/Lecture_08$ ./firstProgram -N=1024
Using 1024 threads = 32 warps
Entry          0, written by thread     0
...
Entry    1046529, written by thread  1023
```

---
class: middle

```
darve@gpu1:~/Lecture_08$ ./firstProgram -N=1025
Using 1025 threads = 33 warps
CUDA error at firstProgram.cu:48 code=9(cudaErrorInvalidConfiguration) 
    "cudaGetLastError()" 
```

!!!