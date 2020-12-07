## 2020-11-11以前
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

## 2020-12-7
### 如何编译
1. 准备好arm_compute-v20.08-bin-android prebuilt binaries，其位置为/path/to/acl，设置环境变量
```sh
export ACL_HOME=/path/to/acl
```
2. 安装ndk（使用r21b版本的），并使用ndk安装standalone-toolchains：
```sh
$NDK/build/tools/make_standalone_toolchain.py --arch arm64 --install-dir $MY_TOOLCHAINS/aarch64-linux-android-ndk-r21b --stl libc++ --api 21
```
这里可能会提示不需要进行安装，因为r21b的ndk似乎自带了toolchains，可以使用$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin中的aarch64-linux-android21-clang++进行交叉编译，把它加入PATH中。
3. 编译：
```sh
make measure_gemm
```
4. 运行：
```sh
adb push <binary> /data/local/tmp
adb shell
cd /data/local/tmp
chmod 777 <binary>
./<binary>
```
5. 输出：
以GEMM为例子，输出如下，最后的sgementation fault目前无法避免，即使是运行官方例子也会有这个问题，但是它应该不影响测性能，似乎是回收资源的时候出了问题。
```sh
HWELS:/data/local/tmp $ ./measure_acl_gemm_aarch64
Time cost of GEMM with shape 16x16x16 dtype= float32 is: 0.06008 ms(100runs).
Time cost of GEMM with shape 32x32x32 dtype= float32 is: 0.05066 ms(100runs).
Time cost of GEMM with shape 64x64x64 dtype= float32 is: 0.05106 ms(100runs).
Time cost of GEMM with shape 128x128x128 dtype= float32 is: 0.10623 ms(100runs).
Time cost of GEMM with shape 256x256x256 dtype= float32 is: 0.4792 ms(100runs).
Time cost of GEMM with shape 512x512x512 dtype= float32 is: 1.67961 ms(100runs).
Time cost of GEMM with shape 1024x1024x1024 dtype= float32 is: 8.97776 ms(100runs).
Segmentation fault
```
6. 更多测试：
测更多的shape，目前需要自己改源码中main函数里的shape信息。