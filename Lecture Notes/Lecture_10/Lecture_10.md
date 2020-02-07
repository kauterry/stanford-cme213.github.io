class: center, middle

# CME 213, ME 339 - Winter 2020

## Eric Darve, ICME

![:width 40%](Stanford.jpg)

“Software is like entropy: It is difficult to grasp, weighs nothing, and obeys the Second Law of Thermodynamics; i.e., it always increases.”
(Norman Augustine)

---
class: center, middle

# Concurrency and latency

---
class: center, middle

Imagine you are a pencil manufacturer.

You outsource your manufacturing plants to China but your market is in the US.

How do you organize the logistics of the transport?

![:width 30%](2020-02-05-13-14-35.png)

---
class: middle

Concurrency is used to hide long latencies:

- memory access
- floating point units
- any long sequence of operations

---
class: center, middle

Processors are optimized in the same way

Hide latency through concurrency

![:width 50%](warp_scheduler.png)