gemm_cpu_naive: gemm_cpu_naive.cpp
	g++ -std=c++11 -o gemm_cpu_naive gemm_cpu_naive.cpp -lOpenCL

gemm_cpu_opt: gemm_cpu_opt.cpp
	g++ -std=c++11 -mavx -o gemm_cpu_opt gemm_cpu_opt.cpp -lOpenCL

gemm_gpu_naive: gemm_gpu_naive.cpp
	g++ -std=c++11 -o gemm_gpu_naive gemm_gpu_naive.cpp -lOpenCL

gemm_gpu_memory_opt: gemm_gpu_memory_opt.cpp
	g++ -std=c++11 -o gemm_gpu_memory_opt gemm_gpu_memory_opt.cpp -lOpenCL

gemm_gpu_vload_opt: gemm_gpu_vload_opt.cpp
	g++ -std=c++11 -o gemm_gpu_vload_opt gemm_gpu_vload_opt.cpp -lOpenCL

gemm_gpu_multi_item_opt: gemm_gpu_multi_item_opt.cpp
	g++ -std=c++11 -o gemm_gpu_multi_item_opt gemm_gpu_multi_item_opt.cpp -lOpenCL

gemm_gpu_pipeline_opt: gemm_gpu_pipeline_opt.cpp
	g++ -std=c++11 -o gemm_gpu_pipeline_opt gemm_gpu_pipeline_opt.cpp -lOpenCL

all: gemm_cpu_naive gemm_cpu_opt gemm_gpu_naive gemm_gpu_memory_opt gemm_gpu_vload_opt gemm_gpu_multi_item_opt gemm_gpu_pipeline_opt

clean:
	rm -rf gemm_cpu_naive gemm_cpu_opt gemm_gpu_naive gemm_gpu_memory_opt gemm_gpu_vload_opt gemm_gpu_multi_item_opt gemm_gpu_pipeline_opt