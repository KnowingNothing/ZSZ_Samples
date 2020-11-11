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


## 2020-11-11
### 如何做arm compute library的测试
>这里的介绍的是使用native android编译，而不是交叉编译，虽然整个流程还是需要host机器做一些文件传输工作
>这里介绍的方法也比较愚笨，因为我不太熟悉android开发
#### 1. 下载arm compute library的prebuilt binary
#### 2. 将本代码仓拷贝到arm compute library里面
假设我们的arm compute library的名字叫arm_compute-v20.08-bin-android
#### 3. 在目标手机上安装termux
同时需要安装好sshd，启动sshd并且与host机器在同一局域网下，这样就能ssh连上，此外还要配置termux能访问storage（sd卡）
#### 4. 编译
把含有本代码仓的arm_compute-v20.08-bin-android压缩后通过scp传给termux，再把/system/vendor/libOpenCL.so拷贝进入arm_compute-v20.08-bin-android。在termux内部用g++进行编译，大概需要以下内容：
```sh
g++ measure_op/measure_acl_gemm.cc utils/*.cpp  -O2 -std=c++11 -I. -Iinclude -Iutils -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android -L/data/data/com.termux/files/home/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o measure_gemm
```
编译的时候直接在arm_compute-v20.08-bin-android里编译，输出也在同一个目录。之后将整个arm_compute-v20.08-bin-android拷贝到sd卡上。此外还要把../usr/lib拷贝到sd卡
#### 5. 运行
在host机器上adb pull从sd卡把arm_compute-v20.08-bin-android拷出来，再adb push到/data/local/tmp里，对于sd卡里的usr/lib也一样，然后adb shell进入/data/local/tmp，进入arm_compute-v20.08-bin-android，运行
```sh
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/arm_compute-v20.08-bin-android/lib/android-arm64-v8a-cl:/data/local/tmp/usr/lib ./measure_gemm
```