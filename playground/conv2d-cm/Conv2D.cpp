/*
 *
 * Author: Size Zheng
 * Date: 2019-11-02
 * Description: The host code for simple 2D convolution
 * Notes: Only 3x3 kernel and 4x4 block now
 * 
*/
#include <iostream>
#include <assert.h>
#include <chrono>
#include <cmath>
#include "cm_rt_helpers.h"

typedef unsigned char BYTE;
typedef enum{RowMajor, ColMajor, Nd} storage_type_t;

#define EXPAND(x, y) ((x + y - 1) & ~(y - 1))
#define CEIL_DIV(x, y) ((x + y - 1) / y)

CmProgram* LoadProgram(CmDevice* pCmDev, char* code)
{
    FILE* pISA = fopen(code, "rb");

    fseek(pISA, 0, SEEK_END);
    int codeSize = ftell(pISA);
    rewind(pISA);

    void *pCommonISACode = (BYTE*) malloc(codeSize);

    fread(pCommonISACode, 1, codeSize, pISA);
    fclose(pISA);

    CmProgram* program = nullptr;
    cm_result_check(pCmDev->LoadProgram(pCommonISACode, codeSize, program));
    free(pCommonISACode);

    return program;
}

int RunConv2D_conv2d_kernel_3x3_b8x4(int H, int W, int R, int S, int gdx, int gdy, int nIter=1)
{
    // check input
    assert(H >= R && W >= S);
    // each thread process 4 x 4 data
    int tdx = 4;
    int tdy = 128;

    storage_type_t stt = RowMajor;

    int H_out = H;
    int W_out = W;
    
    // notice that the row and column order
    // ------------------------> x
    // | (0, 0), (0, 1), (0, 2) ...
    // | (1, 0), (1, 1), (1, 2) ...
    // | (2, 0), (2, 1), (2, 2) ...
    // | (3, 0), (3, 1), (3, 2) ...
    // |  ...     ...     ...
    // v
    // y
    int num_ty = CEIL_DIV(H_out, tdy);
    int num_tx = CEIL_DIV(W_out, tdx);
    int ty = gdy;
    int tx = gdx;
    int gy = CEIL_DIV(num_ty, ty);
    int gx = CEIL_DIV(num_tx, tx);
    // check the group parameters
    // std::cout << gx << " " << gy << " " << tx << " " << ty << std::endl;
    std::cout << "2D convolution example..." << std::endl;
    std::cout << "Image shape is (" << H << ", " << W << ")" << std::endl;
    std::cout << "Kernel size if (" << R << ", " << S << ")" << std::endl;
    std::cout << "Use (" << gy << "x" << gx << ") groups and (" << ty << "x" << tx << ") threads per group" << std::endl; 

    CmDevice* pCmDev = nullptr;
    unsigned int version = 0;
    cm_result_check(::CreateCmDevice(pCmDev, version));
    if(version < CM_1_0)
    {
        std::cout << "Runtime API too old (" << version << "<" << CM_1_0 << ")" << std::endl;
        return -1;
    }

    CmProgram* program = LoadProgram(pCmDev, "Conv2D_genx.isa");

    CmKernel* kernel = nullptr;
    // the kernel name must be identical to that in xxx_genx.cpp
    cm_result_check(pCmDev->CreateKernel(program, "conv2d_kernel_3x3_b8x4", kernel));

    // the input data must be aligned by 16 bytes, for 3x3 convolution, it's sufficient to add 2 floats per row
    float* image_host = (float*) malloc((H + R - 1) * (W + S + 1) * sizeof(float));
    float* filter_host = (float*) malloc(EXPAND(R * S * sizeof(float), 256));

    // init
    for (int i = 0; i < H + R - 1; ++i)
    {
        for (int j = 0; j < W + S + 1; ++j)
        {
            if (i < R / 2 || i >= H + R / 2 || j < S / 2 || j >= W + S / 2)
            {
                image_host[i * (W + S + 1) + j] = 0.0;
            }
            else
            {
                image_host[i * (W + S + 1) + j] = 0.0 + i + j; 
            }
            
        }
    }

    // //check the inputs
    // for (int i = 0; i < H + R - 1; ++i)
    // {
    //     for (int j = 0; j < W + S + 1; ++j)
    //     {
    //         std::cout << image_host[i * (W + S + 1) + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < R; ++i)
    // {
    //     for (int j = 0; j < S; ++j)
    //     {
    //         filter_host[i * S + j] = 0.0 + i + j;
    //     }
    // }

    float* result_host = (float*) malloc(H_out * W_out * sizeof(float));
    float* result_golden = (float*) malloc(H_out * W_out * sizeof(float));

    //TODO: write the kernel
    kernel->SetKernelArg(0, 4, &W);
    int lw = W + S + 1;
    kernel->SetKernelArg(1, 4, &lw);

    CmQueue* pCmQueue = nullptr;
    cm_result_check(pCmDev->CreateQueue(pCmQueue));

    CmBuffer* image = nullptr;
    cm_result_check(pCmDev->CreateBuffer((H + R - 1) * (W + S + 1) * sizeof(float), image));
    CmBuffer* filter = nullptr;
    // notice we have to align the data by 256 bytes, because we want to use SLM
    cm_result_check(pCmDev->CreateBuffer(EXPAND(R * S * sizeof(float), 256), filter));
    CmBuffer* result = nullptr;
    cm_result_check(pCmDev->CreateBuffer(H_out * W_out * sizeof(float), result));

    cm_result_check(image->WriteSurface((BYTE*) image_host, nullptr));
    cm_result_check(filter->WriteSurface((BYTE*) filter_host, nullptr));
    
    SurfaceIndex* image_index = nullptr;
    SurfaceIndex* filter_index = nullptr;
    SurfaceIndex* result_index = nullptr;
    image->GetIndex(image_index);
    filter->GetIndex(filter_index);
    result->GetIndex(result_index);
    
    kernel->SetKernelArg(2, sizeof(SurfaceIndex), image_index);
    kernel->SetKernelArg(3, sizeof(SurfaceIndex), filter_index);
    kernel->SetKernelArg(4, sizeof(SurfaceIndex), result_index);

    CmTask* pKernelArray = nullptr;
    cm_result_check(pCmDev->CreateTask(pKernelArray));
    cm_result_check(pKernelArray->AddKernel(kernel));

    CmThreadGroupSpace* groups = nullptr;
    pCmDev->CreateThreadGroupSpace(gx, gy, tx, ty, groups);

    CmEvent* e = nullptr;
    CM_STATUS s;
    long long kernel_ns = 0;
    long long host_ns = 0;

    for (int it = -1; it < nIter; ++it)
    {
        long long unsigned time_in_ns = 0;
        std::chrono::nanoseconds start = std::chrono::duration_cast<std::chrono::nanoseconds> (
            std::chrono::system_clock::now().time_since_epoch());
        cm_result_check(pCmQueue->EnqueueWithGroup(pKernelArray, e, groups));
        for (e->GetStatus(s); s != CM_STATUS_FINISHED; e->GetStatus(s));
        std::chrono::nanoseconds end = std::chrono::duration_cast<std::chrono::nanoseconds> (
            std::chrono::system_clock::now().time_since_epoch());
        if (it >= 0)
            host_ns += (end.count() - start.count());
        cm_result_check(e->GetExecutionTime(time_in_ns));
        if (it >= 0)
            kernel_ns += time_in_ns;
    }
    double host_time_cost = host_ns / 1e6 / nIter;
    double kernel_time_cost = kernel_ns / 1e6 / nIter;

    cm_result_check(result->ReadSurface((BYTE*) result_host, nullptr));

    for (int i = 0; i < H_out; ++i)
    {
        for (int j = 0; j < W_out; ++j)
        {
            double tmp = 0.0;
            for (int p = 0; p < R; ++p)
            {
                for (int q = 0; q < S; ++q)
                {
                    tmp += image_host[(i + p) * (W + S + 1) + (j + q)] * filter_host[p * S + q];
                }
            }
            result_golden[i * W_out + j] = tmp;
        }
    }

    // check
    int errs = 0;
    for (int i = 0; i < H_out; ++i)
    {
        for (int j = 0; j < W_out; ++j)
        {
            if (fabs(result_golden[i * W_out + j] - result_host[i * W_out + j]) >= 1e-3)
            {
                errs += 1;
            }
        }
    }

    if (errs > 0)
    {
        std::cout << "Results wrong!" << std::endl;
        std::cout << "Print results: (golden, yours):" << std::endl;
        for (int i = 0; i < H_out; ++i)
        {
            for (int j = 0; j < W_out; ++j)
            {
                std::cout << "(" << result_golden[i * W_out + j] << ", " << result_host[i * W_out + j] << ") ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "Passed!" << std::endl;
    }

    std::cout << "Device time cost: " << kernel_time_cost << " ms" <<  std::endl;
    std::cout << "Host time cost  : " << host_time_cost << " ms" << std::endl;
    return -errs;
}


int RunConv2D_conv2d_kernel_3x3(int H, int W, int nIter=1)
{
    CmDevice *pdev = nullptr;
    unsigned int version = 0;
    cm_result_check(::CreateCmDevice(pdev, version));
    if (version < CM_1_0)
    {
        std::cout << "Runtime API too old (" << version << "<" << CM_1_0 << ")" << std::endl;
        return -1;
    }

    CmProgram* program = LoadProgram(pdev, "Conv2D_genx.isa");
    CmKernel* kernel = nullptr;
    cm_result_check(pdev->CreateKernel(program, "conv2d_kernel_3x3", kernel));

    float* image_host = (float*) malloc((H + 2) * (W + 2) * sizeof(float));
    float* filter_host = (float*) malloc(4 * 4 * sizeof(float));

    for (int i = 0; i < (H + 2) * (W + 2); ++i)
    {
        image_host[i] = 0.0f + 1;
    }
    for (int j = 0; j < 4 * 4; ++j)
    {
        filter_host[j] = 0.0f + 2;
    }

    std::cout << "data ready" << std::endl;

    float* result_host = (float*) malloc(H * W * sizeof(float));
    float* result_golden = (float*) malloc(H * W * sizeof(float));

    CmQueue* pCmQueue = nullptr;
    cm_result_check(pdev->CreateQueue(pCmQueue));

    CmSurface2D* image_surface = nullptr;
    cm_result_check(pdev->CreateSurface2D(W + 2, H + 2, CM_SURFACE_FORMAT_A8R8G8B8, image_surface));
    CmSurface2D* filter_surface = nullptr;
    cm_result_check(pdev->CreateSurface2D(4, 4, CM_SURFACE_FORMAT_A8R8G8B8, filter_surface));
    CmSurface2D* result_surface = nullptr;
    cm_result_check(pdev->CreateSurface2D(W, H, CM_SURFACE_FORMAT_A8R8G8B8, result_surface));

    std::cout << "surface ready" << std::endl;

    cm_result_check(image_surface->WriteSurface((BYTE*) image_host, nullptr));

    std::cout << "image write surface ready" << std::endl;
    cm_result_check(filter_surface->WriteSurface((BYTE*) filter_host, nullptr));

    std::cout << "write surface ready" << std::endl;
    
    SurfaceIndex* image_index = nullptr;
    SurfaceIndex* filter_index = nullptr;
    SurfaceIndex* result_index = nullptr;
    image_surface->GetIndex(image_index);
    filter_surface->GetIndex(filter_index);
    result_surface->GetIndex(result_index);
    
    kernel->SetKernelArg(0, sizeof(SurfaceIndex), image_index);
    kernel->SetKernelArg(1, sizeof(SurfaceIndex), filter_index);
    kernel->SetKernelArg(2, sizeof(SurfaceIndex), result_index);

    std::cout << "kernel args ready" << std::endl;

    CmTask* pKernelArray = nullptr;
    cm_result_check(pdev->CreateTask(pKernelArray));
    cm_result_check(pKernelArray->AddKernel(kernel));

    CmThreadSpace* thread_space = nullptr;
    cm_result_check(pdev->CreateThreadSpace(W / 16, H / 8, thread_space));

    CmEvent* e = nullptr;
    CM_STATUS s;
    long long kernel_ns = 0;
    long long host_ns = 0;

    for (int it = -1; it < nIter; ++it)
    {
        long long unsigned time_in_ns = 0;
        std::chrono::nanoseconds start = std::chrono::duration_cast<std::chrono::nanoseconds> (
            std::chrono::system_clock::now().time_since_epoch());
        cm_result_check(pCmQueue->Enqueue(pKernelArray, e, thread_space));
        for (e->GetStatus(s); s != CM_STATUS_FINISHED; e->GetStatus(s));
        std::chrono::nanoseconds end = std::chrono::duration_cast<std::chrono::nanoseconds> (
            std::chrono::system_clock::now().time_since_epoch());
        if (it >= 0)
            host_ns += (end.count() - start.count());
        cm_result_check(e->GetExecutionTime(time_in_ns));
        if (it >= 0)
            kernel_ns += time_in_ns;
    }
    double host_time_cost = host_ns / 1e6 / nIter;
    double kernel_time_cost = kernel_ns / 1e6 / nIter;

    cm_result_check(result_surface->ReadSurface((BYTE*) result_host, nullptr));

    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            double tmp = 0.0;
            for (int p = 0; p < 3; ++p)
            {
                for (int q = 0; q < 3; ++q)
                {
                    tmp += image_host[(i + p) * (W + 2) + (j + q)] * filter_host[p * 4 + q];
                }
            }
            result_golden[i * W + j] = tmp;
        }
    }

    // check
    int errs = 0;
    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            if (fabs(result_golden[i * W + j] - result_host[i * W + j]) >= 1e-3)
            {
                errs += 1;
            }
        }
    }

    if (errs > 0)
    {
        std::cout << "Results wrong!" << std::endl;
        std::cout << "Print results: (golden, yours):" << std::endl;
        for (int i = 0; i < H; ++i)
        {
            for (int j = 0; j < W; ++j)
            {
                std::cout << "(" << result_golden[i * W + j] << ", " << result_host[i * W + j] << ") ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "Passed!" << std::endl;
    }

    std::cout << "Device time cost: " << kernel_time_cost << " ms" <<  std::endl;
    std::cout << "Host time cost  : " << host_time_cost << " ms" << std::endl;
    return -errs;
}


int main(int argc, char** argv)
{
    // gdx must be factor is W && W / gdx >= 4
    // gdy must be factor of H && H / gdx >= 4
    // R and S are fixed because we only have kernel for 3x3
    // H and W should be 2^x where x >= 2
    // size of H and W are limited by SLM && registers, because currently we compute the whole image at a time
    // by spliting the image to small parts, H and W can be unlimited

    // for (int i = 2; i <= 128 / 4; i *= 2)
    // for (int j = 1; j <= 128 / 128; j *= 2)
    //     RunConv2D_conv2d_kernel_3x3_b8x4(/*H=*/128, /*W=*/128, /*R=*/3, /*S=*/3, /*gdx=*/i, /*gdy=*/j, 10);
    RunConv2D_conv2d_kernel_3x3(3200, 3200, 100);
}
