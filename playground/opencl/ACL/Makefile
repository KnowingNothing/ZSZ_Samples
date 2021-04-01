CXX=aarch64-linux-android21-clang++
ACL=$(ACL_HOME)
INCLUDES=-I$(ACL)/ -I$(ACL)/include
LIBS=-L$(ACL)/lib/android-arm64-v8a-cl -larm_compute-static -larm_compute_core-static
FLAGS=-std=c++11 -static-libstdc++ -pie # -DARM_COMPUTE_CL

measure_gemm: measure_acl_gemm.cc $(ACL)/utils/Utils.cpp
	$(CXX) $^ $(INCLUDES) $(FLAGS) $(LIBS) -o $@_aarch64

measure_conv2d: measure_acl_conv2d.cc $(ACL)/utils/Utils.cpp
	$(CXX) $^ $(INCLUDES) $(FLAGS) $(LIBS) -o $@_aarch64

measure_depthwise: measure_acl_depthwise.cc $(ACL)/utils/Utils.cpp
	$(CXX) $^ $(INCLUDES) $(FLAGS) $(LIBS) -o $@_aarch64

clean:
	rm measure_gemm_aarch64 measure_conv2d_aarch64 measure_depthwise_aarch64