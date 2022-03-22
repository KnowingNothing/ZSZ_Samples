#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

int main(int argc, char** argv)
{
    std::cout << "Test 1:" << std::endl;
    thrust::host_vector<double> H(100, 0.0);
    thrust::fill(H.begin() + 10, H.begin() + 80, 1.0);
    thrust::device_vector<double> D = H;
    thrust::host_vector<double> H2(100, -1.0);
    D[0] = 10000.0;
    thrust::copy(D.begin(), D.end(), H2.begin());
    for (int i = 0; i < H2.size(); ++i)
    {
        std::cout << H2[i] << " ";
        if (i % 10 == 9)
        {
            std::cout << std::endl;
        }
    }

    std::cout << "Test 2:" << std::endl;
    size_t N = 10;
    int * raw_ptr;
    cudaMalloc((void **) & raw_ptr, N * sizeof(int));
    thrust::device_ptr<int> dev_ptr(raw_ptr);
    thrust::fill(dev_ptr, dev_ptr + N, (int) 0);
    for (size_t i = 0; i < N; ++i)
    {
        std::cout << *(dev_ptr + i) << " ";
    }
    std::cout << std::endl;

    std::cout << "Test 3:" << std::endl;
    

    return 0;
}