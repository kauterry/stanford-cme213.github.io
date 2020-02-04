class: center, middle

# CME 213, ME 339 - Winter 2020

## Eric Darve, ICME

![:width 40%](Stanford.jpg)

“You can either have software quality or you can have pointer arithmetic, but you cannot have both at the same time.”
(Bertrand Meyer)

---
class: center, middle

GPU Optimization

![:width 30%](2020-01-31-11-33-07.png)

---
class: center, middle

Optimize data transfer from GPU memory

---
class: middle

- Caches are used to optimize memory accesses: L1 and L2 caches.
- Cache behavior is complicated and depends on the compute capability of the GPU.
- We will focus on `sm_37`

---
class: center, middle

# L1 cache

Used primarily for local memory, including temporary register spills

---
class: center, middle

# L2 cache

Cache accesses to local and global memory

---
class: center, middle

The smallest size for a memory transaction is 32 bytes.

That's 8 floats!

---
class: middle

Let's make this concrete with a code

```
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  if (xid < n)
      odata[xid] = idata[xid];
```

---
class: center, middle

Warp requests several memory addresses

These are translated into cache line requests (with a granularity of 32 bytes)

Memory requests are serviced 

---
class: center, middle

![:width 60%](examples-of-global-memory-accesses.png)

---
class: middle

Benchmark: offset access

```
int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
if (xid < n)
    odata[xid] = idata[xid];
```

---
class: middle

```
Elapsed time for offset   0 in msec:     4.5359
Elapsed time for offset   1 in msec:     6.8373
Elapsed time for offset   2 in msec:     6.8377
Elapsed time for offset  32 in msec:     4.5360
Elapsed time for offset  64 in msec:     4.5384
Elapsed time for offset 128 in msec:     4.5398
```

---
class: middle

Benchmark: strided access

```
int xid = stride * (blockIdx.x * blockDim.x + threadIdx.x);
if (xid < n)    
    odata[xid] = idata[xid];
```

---
class: middle

```
Elapsed time for stride   1 in msec:     1.9950
Elapsed time for stride   2 in msec:     2.7842
Elapsed time for stride   4 in msec:     4.0088
Elapsed time for stride   8 in msec:     6.5526
Elapsed time for stride  16 in msec:     7.4371
```

---
class: center, middle

![:width 100%](2020-01-31-13-05-06.png)