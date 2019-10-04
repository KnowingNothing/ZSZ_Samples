## Results of benchmarks

**Device**: NVIDIA TITAN X (Pascal) 
**CUDA**: 10.0
**CUDNN**: 7.6.3
**Tensorflow**: build from source r1.14
**PyTorch**: nightly build v1.3.0
**TVM**: v0.6dev
400 trials with 1 warm-up, mean end-to-end time

| Option | Network | Batch | Time cost |
| ------ | ------- | ----- | --------- |
| pytorch | mobilenet-v1 | 1 |  2.971ms |
| pytorch | mobilenet-v1 | 8 |  6.162ms |
| pytorch | mobilenet-v1 | 32 |  22.902ms |
| pytorch | mobilenet-v1 | 64 |  44.873ms |
| pytorch | mobilenet-v1 | 128 |  inf ms |
| tensorflow | mobilenet-v1 | 1 | 1.834ms |
| tensorflow | mobilenet-v1 | 8 | 6.801ms |
| tensorflow | mobilenet-v1 | 32 | 25.378ms |
| tensorflow | mobilenet-v1 | 64 | 50.104ms |
| tensorflow | mobilenet-v1 | 128 | 100.306ms |
| tensorflow-xla | mobilenet-v1 | 1 | 2.025ms |
| tensorflow-xla | mobilenet-v1 | 8 | 5.132ms |
| tensorflow-xla | mobilenet-v1 | 32 | 16.681ms |
| tensorflow-xla | mobilenet-v1 | 64 | 32.115ms |
| tensorflow-xla | mobilenet-v1 | 128 | 63.321ms |
| tvm-nnvm | mobilenet-v1 | 1 | 0.883ms |
| tvm-nnvm | mobilenet-v1 | 8 | 3.504ms |
| tvm-nnvm | mobilenet-v1 | 32 | 12.096ms |
| tvm-nnvm | mobilenet-v1 | 64 | 23.345ms |
| tvm-nnvm | mobilenet-v1 | 128 | 46.021ms |
| tvm-nnvm | mobilenet-v2 | 1 | 1.788ms |
