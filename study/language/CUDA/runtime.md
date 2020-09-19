1. runtime是在cudart库里实现的
2. 所有的entry point的开头都是cuda
3. runtime的初始化是隐式的，在第一次调用runtime API的时候自动完成
4. initialization 时候为每个device创建一个context，同时进行必要的JIT
5. cudaDeviceReset()会销毁当前操作的device的context，下次runtime API调用会再创建
6. device memory可以申请为linear memory或者CUDA arrays
7. CUDA array和texture fetching有关，linear memory则是使用40-bit的地址空间寻址
8. linear memory可以使用cudaMalloc()申请，用cudaFree()释放，用cudaMemcpy()传输
9. cudaMallocPitch(), cudaMalloc3D()
10. shared memory 用__shared__指明