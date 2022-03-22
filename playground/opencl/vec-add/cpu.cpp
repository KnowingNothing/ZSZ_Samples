// This program implements a vector addition using OpenCL

// System includes
#include <chrono>
#include <iostream>

void compute_kernel(const int* A, const int* B, int* C, int elements) {
    for (int i = 0; i < elements; ++i) {
        C[i] = A[i] + B[i];
    }
}

void try_with(int elements) {
    // This code executes on the OpenCL host
    
    // Host data
    int *A = NULL;  // Input array
    int *B = NULL;  // Input array
    int *C = NULL;  // Output array
    
    // Compute the size of the data 
    size_t datasize = sizeof(int)*elements;

    // Allocate space for input/output data
    A = (int*)malloc(datasize);
    B = (int*)malloc(datasize);
    C = (int*)malloc(datasize);
    // Initialize the input data
    for(int i = 0; i < elements; i++) {
        A[i] = i;
        B[i] = i;
    }

    int repeat = 20;
    compute_kernel(A, B, C, elements);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        compute_kernel(A, B, C, elements);
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "CPU serial cost for elements=" << elements << " is " << time_span.count() / repeat * 1e3 << " ms.\n";

    // Free host resources
    free(A);
    free(B);
    free(C);
}

int main() {
    for (int i = 256; i <= 8192; i = i * 2) {
        try_with(i);
    }
}
