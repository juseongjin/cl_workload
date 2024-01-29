<h1>Matrix Multiplication based on OpenCL</h1>

1. OpenGL/OpenCL setting & Install
2. How to use
3. Considering

   
## Install
OpenGL|EGL environment setting
```
$ sudo apt update
```
Arm GPU(Mali t62x) hardware driver install
```
$ sudo apt install mali-t62x-wayland-driver
```
For using graphics interface library such as OpenGL|ES
```
$ sudo apt install libegl1-mesa-dev libgles2-mesa-dev
```
OpenCL environment setting
기존의 OpenCL 관련 library 삭제
```
$ sudo apt purge --autoremove ocl-icd-libopencl1 opencl-headers
```
clinfo install (cluster information service), SNMP monitor 기능 제공
```
$ sudo apt install clinfo
```
```
$ sudo mkdir /etc/OpenCL
```
clinfo command 사용 시, GPU 정보를 읽어오기 위해 필요한 작업
```
$ sudo mkdir /etc/OpenCL/vendors
```
```
$ sudo sh -c 'echo "/usr/lib/arm-linux-gnueabihf/mali-egl/libOpenCL.so" > /etc/OpenCL/vendors/armocl.icd'
```
CL/CL header library 설치
```
$ sudo apt install ocl-icd-opencl-dev opencl-headers
```
## How to use
clone this repository 
```
$ git clone git@github.com:juseongjin/cl_workload.git
```
```
$ cd ~/cl_workload
```
```
$ make
```
```
$ ./matrix_multiply <duration> 0 <GPU thread num> 0
```
![Screenshot from 2024-01-29 15-39-06](https://github.com/juseongjin/cl_workload/assets/49185122/a1aea496-5835-49ad-8fdb-f687833571a5)

## Considering
In matrix_multiply.cc file, GPU_MAT_SIZE is a kernel size. Matrix size = GPU_MAT_SIZE * GPU_MAT_SIZE
If you want to increase GPU contention, you have to change to GPU_MAT_SIZE.
The GPU_MAT_SIZE must be <=(under) 512. Because, GPU_MAT_SIZE > 512 cause overflow and GPU calculation error.
