IMKL = /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/include
LMKL = /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64
LCOMPILER = /opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/
CFLAGS = -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

dgemm:
	gcc -I $(IMKL) -L $(LMKL) -L $(LCOMPILER) $(CFLAGS) dgemm_example.c -o dgemm

run: dgemm
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(LMKL):$(LCOMPILER) ./dgemm