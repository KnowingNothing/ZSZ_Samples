# compile
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl:/system/vendor/lib64/
g++ measure_op/measure_acl_gemm.cc utils/*.cpp  -O2 -std=c++11 -I. -Iinclude -Iutils -L/system/vendor/lib64/ -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o measure_gemm
```
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl:/data/data/com.termux/files/home/arm_compute-v20.08-bin-android
g++ measure_op/measure_acl_gemm.cc utils/*.cpp  -O2 -std=c++11 -I. -Iinclude -Iutils -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o measure_gemm
```
```sh
g++ cl_sgemm.cpp utils/*.cpp  -O2 -std=c++11 -I. -Iinclude -Iutils -L/system/vendor/lib64 -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o cl_sgemm

g++ cl_sgemm.cpp utils/*.cpp  -O2 -std=c++11 -I. -Iinclude -Iutils -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o cl_sgemm
```

g++ measure_op/measure_acl_conv2d.cc utils/*.cpp  -O2 -std=c++11 -I. -Iinclude -Iutils -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o measure_conv2d