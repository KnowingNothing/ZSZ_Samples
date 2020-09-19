1. 一个kernel由N个thread执行N次
2. 一个kernel被__global__修饰
3. 有多少线程执行kernel通过<<<>>>控制
4. 对thread的编号来说，最多由3维id控制，(x+y*Dx+z*Dx*Xy)，线程以block组织
5. 一个线程块最多1024个线程
6. 线程块以grid组织，block的编号也是最多三维
7. grid的维度大小由处理的数据大小和GPU拥有的processor为限
8. 这个guide里提到了thread block大小为16x16是常用选择
9. 同一个block内部的线程可以共享shared memory，以及用同步原语，如__syncthreads()，还有很多其它原语
10. shared memory类似L1 cache
11. 每个线程有private local memory, 每个block有shared memory，所有线程可以访问global memory, constant memory, texture memory
12. 多个kernel launch之间，global, constant, texture memory都是不变的
13. compute capability: SM version, decide which hardware features, instructions are available
14. X.Y: X一样，core architecture一样， 1: Tesla 2: Fermi 3: Kepler 5: Maxwell 6: Pascal 7: Volta
15. 任何使用了cuda extension的程序都得通过nvcc编译，此外cuda还提供了runtime用于处理数据传输和多设备管理，在runtime底层是driver API，用户也可以用到，在driver API里有两个数据结构，context（类似进程）和module（类似动态链接库）
16. cuda指令集是PTX，其上层是C，无论用哪个写的，都得通过nvcc编译
17. nvcc的离线(offline)编译流程：
  1. 分开device和host代码
  2. 把device代码编译为PTX或者cubin
  3. 同时把host代码里的<<<>>>替换为对PTX或者cubin的load与launch API
  4. 修改后的Host代码可以直接用gcc编译，这个过程可以手动，也可以由nvcc自己调用完成
18. 对于应用来说，有两种方式调用
  1. 把nvcc编译好的host代码链接
  2. 使用cuda runtime API load和execute PTX或者cubin
19. 如果应用选择了load PTX去执行，会触发driver的JIT机制
20. nvcc指定cubin的兼容性：-code=sm_35
21. nvcc指定PTX的兼容性：-arch=compute_30
22. 在device code里还可以看到宏__CUDA_ARCH__，比如是350
23. 生成的device code的指针宽度：-m32或者-m64决定（如果nvcc的位数本身就是正确的，不用指定-m）
  


需要看的manual
CUDA reference manual
C Language Extensions
Driver API
nvcc user manual