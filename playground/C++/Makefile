IMKL = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/include
LMKL = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
LCOMPILER = /opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/
CFLAGS = -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -fopenmp -mavx512f

gemm_hybrid_intel: gemm_hybrid.cc
	g++ -I $(IMKL) -L $(LMKL) -L $(LCOMPILER) $(CFLAGS) gemm_hybrid.cc -o gemm_hybrid_intel

run_intel: gemm_hybrid.cc gemm_hybrid_intel
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(LMKL):$(LCOMPILER) ./gemm_hybrid_intel


hybrid_inst: hybrid_inst.cc
	g++ -fopenmp -mavx512f $^ -o $@