# How to know if the kernel is using tensor core?
1. use nsight command line interface:
```sh
nv-nsight-cu-cli --csv --metrics sm__inst_executed_pipe_tensor_op_hmma.sum <binary> 2>&1 | tee profile.csv
```
```sh
nv-nsight-cu-cli --csv --metrics sm__inst_executed_pipe_tensor_op_imma.sum <binary> 2>&1 | tee profile.csv
```
This metric will tell how many hmma instructions are emitted during execution.
> reference:
    https://developer.nvidia.com/blog/using-nsight-compute-nvprof-mixed-precision-deep-learning-models/

```sh
CUDA_VISIBLE_DEVICES=0 nvprof --metrics tensor_precision_fu_utilization,tensor_int_fu_utilization ./cudnn_conv_benchmark
```
这是nvprof的命令。