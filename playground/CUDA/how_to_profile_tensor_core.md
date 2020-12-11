# How to know if the kernel is using tensor core?
1. use nsight command line interface:
```sh
nv-nsight-cu-cli --csv --metrics sm__inst_executed_pipe_tensor_op_hmma.sum <binary> 2>&1 profile.csv
```
This metric will tell how many hmma instructions are emitted during execution.
> reference:
    https://developer.nvidia.com/blog/using-nsight-compute-nvprof-mixed-precision-deep-learning-models/