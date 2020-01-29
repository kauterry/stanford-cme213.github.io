---
layout: default
title: "CME 213: Introduction to Parallel Computing using MPI, openMP, and CUDA"
---

Welcome to CME 213!

This class will give hands-on experience with programming multicore processors, graphics processing units (GPU), and parallel computers. The focus will be on the message passing interface (MPI, parallel clusters) and the compute unified device architecture (CUDA, GPU). Topics will include multithreaded programs, GPU computing, computer cluster programming, C++ threads, OpenMP, CUDA, and MPI. Pre-requisites include C++, templates, debugging, UNIX, makefile, numerical algorithms (differential equations, linear algebra).

### Lecture notes and code
* [Lecture 1](Lecture Notes/Lecture_01/Lecture_01.html)
* [Lecture 2](Lecture Notes/Lecture_02/Lecture_02.html)
* [C++ Refresher](Tutorial Notes/Tutorial_01.pdf) and [code](Tutorial Notes/code.zip)
* [Lecture 3](Lecture Notes/Lecture_03/Lecture_03.html) and [code](https://github.com/stanford-cme213/stanford-cme213.github.io/tree/master/Code/Lecture_03)
* [Lecture 4](Lecture Notes/Lecture_04/Lecture_04.html) and [code](https://github.com/stanford-cme213/stanford-cme213.github.io/tree/master/Code/Lecture_04)
* [Lecture 5](Lecture Notes/Lecture_05/Lecture_05.html) and [code](https://github.com/stanford-cme213/stanford-cme213.github.io/tree/master/Code/Lecture_05)
* [Lecture 6](Lecture Notes/Lecture_06/Lecture_06.html) and [code](https://github.com/stanford-cme213/stanford-cme213.github.io/tree/master/Code/Lecture_06)

### Videos
* [Lecture 1](https://youtu.be/mmJ__Ygexos)
* [Lecture 2](https://youtu.be/ETOp2NkZrlo)
* [Lecture 3](https://youtu.be/Mi1vwbsk6Fw)
* [Lecture 4](https://youtu.be/wN0JRtCp7YY)
* [Lecture 5](https://youtu.be/Oip9nrDzAAQ)
* [Lecture 6](https://youtu.be/O8v4JXwDGws)

### Office Hours
* Wednesday 7-9PM; TA led; Huang 304
* Thursday 7-9PM; TA led; Huang 304
* Thursday 9-10:45AM; Instructor led; bldg 520, room 120 

### Homework
* [Homework 1](Homework/hw1.pdf) and [starter code](Homework/code_hw1.zip)
* [Homework 2](Homework/hw2.pdf) and [Radix Sort Tutorial](Homework/RadixSortTutorial.pdf) and [starter code](Homework/code_hw2.zip)

### Google Cloud Platform 
* [Google Cloud Platform setup instructions](./gcp.html)
* [VM instances information page](https://console.cloud.google.com/compute)
* [Billing page](https://console.cloud.google.com/billing)
* [GCP dashboard](https://console.cloud.google.com/home)

### Class Information
* Location: [School of Education, Room 128](https://campus-map.stanford.edu/?srch=School+of+Education+128)  
* Lectures: Mon/Wed/Fri 1:30-2:50PM
* Instructor: [Eric Darve](https://me.stanford.edu/people/eric-darve)  
* Course Assistants: William Jen, Kingway Liang
* [Syllabus](https://canvas.stanford.edu/courses/110850/assignments/syllabus)

### Reading and links

#### OpenMP
* [OpenMP LLNL guide](https://computing.llnl.gov/tutorials/openMP/)
* [OpenMP guide by Yliluoma](https://bisqwit.iki.fi/story/howto/openmp/)
* [OpenMP 5.0 Reference Guide](https://www.openmp.org/wp-content/uploads/OpenMPRef-5.0-0519-web.pdf)
* [OpenMP API Specification](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5.0.pdf)

#### C++ threads
* [C++ reference](https://en.cppreference.com/w/cpp)
* [Simple examples of C++ multithreading](https://www.geeksforgeeks.org/multithreading-in-cpp/)
* [C++ threads](https://en.cppreference.com/w/cpp/thread/thread/thread)
* [LLNL tutorial on Pthreads](https://computing.llnl.gov/tutorials/pthreads/)

#### CUDA
* [CUDA Programming Guides and References](http://docs.nvidia.com/cuda/index.html)
* [CUDA C++ Programming Guide](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
* [CUDA C++ Best Practices Guide](http://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
* [CUDA compiler driver NVCC](https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf)
* [OpenACC](https://www.openacc.org/)
* [Compilers that support OpenACC](https://www.openacc.org/tools)
* [OpenACC Specification (ver. 2.7 November 2018)](https://www.openacc.org/sites/default/files/inline-files/OpenACC.2.6.final.pdf)
* [OpenACC Programming and Best Practices Guide](http://www.openacc.org/sites/default/files/OpenACC_Programming_Guide_0.pdf)
* [OpenACC 2.7 API Reference Card](https://www.pgroup.com/lit/literature/openacc-api-guide-2.7.pdf)
* [CUDA occupancy calculator](Reading/CUDA_Occupancy_Calculator.xls)
* [Kepler GK110/210 whitepaper](Reading/NVIDIA_Kepler_GK110_GK210_Architecture_Whitepaper.pdf) (the K80 uses the GK210 chip)
* [Data sheet for Tesla GPUs](https://en.wikipedia.org/wiki/Nvidia_Tesla). **Tesla** is NVIDIA's brand name for their products targeting stream processing or general-purpose graphics processing units (GPGPU).
* [K80 device info](k80.md)
* [K80 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/TeslaK80-datasheet.pdf)

#### Sorting algorithms
* [A novel sorting algorithm for many-core architectures based on adaptive bitonic sort](https://ieeexplore.ieee.org/abstract/document/6267838)
* [Adaptive Bitonic Sorting](https://pdfs.semanticscholar.org/bcdf/c4e40c79547c9daf89dada4e1c23056871cb.pdf)
