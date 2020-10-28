## CUBLAS GEMM接口的几个注意

GEMM这个运算人尽皆知，cuBLAS也直接提供了接口，但是发现用的时候容易弄错参数的传递。cuBLAS有自己的一套约定，比如传入矩阵默认column major。但是文档上一下子也不好找到相关叙述，这里是通过几个测试确定的GEMM的数学接口，记下来方便以后对照。

首先GEMM的接口大概这样调用：
```c++
cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              MATRIX_M, MATRIX_N, MATRIX_K,
                              &alpha,
                              a_fp16, CUDA_R_16F, MATRIX_M,
                              b_fp16, CUDA_R_16F, MATRIX_K,
                              &beta,
                              c_cublas, CUDA_R_32F, MATRIX_M,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                            ));
```
我们认为是完成$C = \alpha \times A*B + \beta \times C$的运算，这里MATRIX_M就是指A矩阵的行数（逻辑上），MATRIX_M就是B矩阵的列数，MATRIX_K就是A的列数和B的行数。我们假设正常写一个大写字母就表示一个row major的矩阵，一个column major的矩阵则是row major矩阵的转置。在传入参数时每个矩阵还对应传入了一个leading dimension的长度，这就是当前layout下连续维度的长度（比如row major的列数和column major的行数）。而CUBLAS_OP_N控制是否再进行转置，如果要转置，这个值就是CUBLAS_OP_T。

知道了这些，上述例子，我们传入了两个column major的矩阵，就相当于传入了$A^T$和$B^T$，最终得到的C也要column major的，所以三个矩阵的leading dimension都传入行数，这样我们算出来的结果是$C^T$，也就相当于$B^T A^T$，所以GEMM接口总可以理解为第二个矩阵右乘第一个矩阵，至于其它参数都看需求更改。比如我们想输入$A$和$B$（也就是都row major），得到$C$（也是row major），那我们就可以把$B$第一个传入，配上leading dimension为列数，$A$第二个传入，leading dimension也是列数，$C$放在最后，leading dimension也是行数，如下：
```c++
cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              MATRIX_N, MATRIX_M, MATRIX_K,
                              &alpha,
                              b_fp16, CUDA_R_16F, MATRIX_N,
                              a_fp16, CUDA_R_16F, MATRIX_K,
                              &beta,
                              c_cublas, CUDA_R_32F, MATRIX_N,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                            ));
```
注意传入的时候各个位置的参数都要对应好。

再举一个例子，如果我们想输入$A$和$B$（也就是都row major），得到$C^T$（是column major），我们就可以用如下传入：
```c++
cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              MATRIX_M, MATRIX_N, MATRIX_K,
                              &alpha,
                              a_fp16, CUDA_R_16F, MATRIX_K,
                              b_fp16, CUDA_R_16F, MATRIX_N,
                              &beta,
                              c_cublas, CUDA_R_32F, MATRIX_M,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                            ));
```
这里利用了CUBLAS_OP_T增加一次转置，相当于运算$(B)^T(A)^T = (AB)^T = C^T$

再举一个例子，如果我们想输入$A$（是row major）和$B^T$（是都column major），得到$C^T$（是column major），我们就可以用如下传入：
```c++
cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                              MATRIX_M, MATRIX_N, MATRIX_K,
                              &alpha,
                              a_fp16, CUDA_R_16F, MATRIX_K,
                              b_fp16, CUDA_R_16F, MATRIX_K,
                              &beta,
                              c_cublas, CUDA_R_32F, MATRIX_M,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                            ));
```
这里利用了CUBLAS_OP_T增加一次转置，相当于运算$(B^T)(A)^T = (AB)^T = C^T$