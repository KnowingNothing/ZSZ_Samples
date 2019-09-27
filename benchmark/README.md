## Results of benchmarks

**Device**: NVIDIA TITAN X (Pascal) 
**CUDA**: 10.0
**CUDNN**: 7.6.3
**Tensorflow**: build from source r1.14
**PyTorch**: nightly build v1.3.0
**TVM**: v0.6dev

| Option | Network | Batch | Time cost |
| ------ | ------- | ----- | --------- |
| pytorch | mobilenet-v1 | 1 |  3.14ms |
| pytorch | mobilenet-v1 | 8 |  6.46ms |
| pytorch | mobilenet-v1 | 32 |  21.72ms |
| pytorch | mobilenet-v1 | 64 |  40.77ms |
| pytorch | mobilenet-v1 | 128 |  inf ms |
| tensorflow | mobilenet-v1 | 1 | 2.827ms |
| tensorflow | mobilenet-v1 | 8 | 7.10ms |
| tensorflow | mobilenet-v1 | 32 | 23.43ms |
| tensorflow | mobilenet-v1 | 64 | 46.74ms |
| tensorflow | mobilenet-v1 | 128 | 89.43ms |
| tensorflow-xla | mobilenet-v1 | 1 | 3.90ms |
| tensorflow-xla | mobilenet-v1 | 8 | 4.328ms |
| tensorflow-xla | mobilenet-v1 | 32 | 15.23ms |
| tensorflow-xla | mobilenet-v1 | 64 | 25.70ms |
| tensorflow-xla | mobilenet-v1 | 128 | 52.42ms |
| tvm-nnvm | mobilenet-v1 | 1 | 0.947ms |
| tvm-nnvm | mobilenet-v1 | 8 | 3.68ms |
| tvm-nnvm | mobilenet-v1 | 32 | 11.94ms |
| tvm-nnvm | mobilenet-v1 | 64 | 22.88ms |
| tvm-nnvm | mobilenet-v1 | 128 | 45.14ms |
| tvm-nnvm | mobilenet-v2 | 1 | 1.788ms |
