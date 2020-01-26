class: center, middle

# CME 213, ME 339 - Winter 2020

## Eric Darve, ICME

![:width 40%](Stanford.jpg)

“If debugging is the process of removing bugs, then programming must be the process of putting them in.”
(Edsger W. Dijkstra)

---
class: middle, center

Before we start...

Download from `Code/`

[Code/create_vm_gpu1.sh](https://github.com/stanford-cme213/stanford-cme213.github.io/blob/master/Code/create_vm_gpu1.sh)

Run 

```
$ ./create_vm_gpu1.sh
```

on your laptop.

---
class: middle

CME 213 so far:

- C++ threads
- OpenMP: `for` loop and `task`
- Sorting algorithms on shared memory

Onwards to GPU computing!

---
class: middle, center

![:width 80%](42-years-processor-trend.png)

---
class: middle, center

Reference

https://github.com/karlrupp/cpu-gpu-mic-comparison

https://www.karlrupp.net/2013/06/cpu-gpu-and-mic-hardware-characteristics-over-time/

https://www.karlrupp.net/2015/06/40-years-of-microprocessor-trend-data/

---
class: middle, center

![:width 80%](2020-01-25-16-54-58.png)

---
class: middle, center

![:width 80%](2020-01-25-16-55-29.png)

---
class: middle, center

![:width 80%](2020-01-25-16-55-49.png)

---
class: middle, center

![:width 80%](2020-01-25-16-56-06.png)

---
class: middle, center

![:width 80%](2020-01-25-16-57-47.png)

---
class: img-right

![](v100.jpg)

# Example: Volta V100

- 8.2 teraflops double-precision performance
- 16.4 teraflops single-precision performance
- 130 teraflops for tensor (deep learning)
- 1134 GB/sec memory bandwidth
- 250 Watts power

---
class: middle, center

![:width 80%](v100-perf.png)

---
class: middle, center

![:width 80%](v100-dl.png)

---
class: middle, center

# What is the technology behind GPU processors?

