# cutlass阅读记录

在一个GPU上去实现一个GEMM，需要什么？
首先，GEMM的运算信息是需要的，包括A, B, C的data type和layout，但是不用给shape，GEMM本身算法是和shape无关的。
但是这些还不足以帮助我们搞清楚如何在一个真正的硬件上实现GEMM，我们还需要有关体系结构的信息，看cutlass的device/gemm.h，可以在Gemm这个类的模板中看到架构信息。总结而言：
1. 算法信息：A, B, C的dtype和layout, IsBetaZero
2. 架构信息：累加时的dtype，OperatorClass, ArchTag, ThraedblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages, AlignmentA, AlignmentB, SplitKSerial, Operator
其中，累加dtype, OperatorClass和ArchTag决定了除了SplitKSerial，ThreadblockSwizzle以外的其它架构信息，通过DefaultGemmConfiguration

在Gemm内部，有一个GemmKernel，它是DefaultGemm内部的GemmKernel，由上述所有算法和架构信息共同决定，它自己本身是一个Operator，内部有Params和SharedStorage

Gemm内部还有一个Arguments结构，是方便传入要解决的实际GEMM问题实例的信息，包括problem_size，三个tensor的ref，ref_D, epilogue(alpha和beta信息)，split_k_slices

Gemm实例会有一个私有的params_，这个就是GemmKernel的Params，每次Gemm进行init时候，都会根据传入的Arguments来修改params_。

每次调用()运算符，如果传入了新的Arguments，就会重新init，init除了设置params_，还会分配workspace，确定grid的设置（根据problem_size和ThreadblockShape）。
之后就会调用run，这个run会根据ThreadblockSwizzle确定最终grid的设置，至于block的设置，是GemmKernel里配置的，(GemmKernel::kThreadCount, 1, 1)。然后从GemmKernel::SharedStorage确定shared memory size，根据这个size看是否用cudaFuncAttributeMaxDynamicSharedMemorySize(如果超出了48KB，这个是比较老的配置了)，然后还会设置cudaFuncAttributePreferredSharedMemoryCarveout为100%。
最后，launch GemmKernel（<<<grid, block, dynamic_shared_memory_size, stream>>>那种），通过cutlass::Kernel这个封装（其内部是先给shared_memory，然后调用括号运算符真正调用kernel）

可以看到，使用哪些kernel是由DefaultGemmConfiguration和DefaultGemm决定。

---

再来看kernel下的gemm.h，内部对于Gemm的抽象已经看不到算法信息，就是只有架构信息：Mma, Epilogue, ThreadblockSwizzle_, SplitKSerial

内部的WarpCount是Mma::WarpCount，kThreadCount是32*WarpCount::kCount

它内部的Params抽象是
```c++
cutlass::gemm::GemmCoord problem_size;
cutlass::gemm::GemmCoord grid_tiled_shape;
typename Mma::IteratorA::Params params_A; // ref_A.layout()
typename Mma::IteratorA::TensorRef ref_A; // ref_A
typename Mma::IteratorB::Params params_B; // ref_B.layout()
typename Mma::IteratorB::TensorRef ref_B; // ref_B
typename Epilogue::OutputTileIterator::Params params_C; // ref_C.layout()
typename Epilogue::OutputTileIterator::TensorRef ref_C; // ref_C
typename Epilogue::OutputTileIterator::Params params_D; // ref_D.layout()
typename Epilogue::OutputTileIterator::TensorRef ref_D; // ref_D
typename OutputOp::Params output_op; // args.epilogue
int* semaphore;  // static_cast<int*>(workspace)
int gemm_k_iterations; // 这个好像根本就没设置，代码写错了吧
int gemm_k_size;  // 一个thread block要做多长的k reduction（gemm_k_iterations * Mma::Shape::kK）
```

另一个抽象是SharedStorage
```c++
union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
};
```