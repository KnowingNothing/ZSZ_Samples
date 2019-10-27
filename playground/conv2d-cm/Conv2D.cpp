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

int RunConv2D(int H, int W, int R, int S, int gdx, int gdy, int KH=1024, int KW=1024, int nIter=1)
{
    // check input
    assert(H >= R && W >= S);
    // each thread process 16 x 16 data
    int tdx = 16;
    int tdy = 16;

    storage_type_t stt = RowMajor;

    int H_out = (H - R) + 1;
    int W_out = (W - S) + 1;
    
    int num_ty = CEIL_DIV(H_out, tdy);
    int num_tx = CEIL_DIV(W_out, tdx);
    int ty = gdy;
    int tx = gdx;
    int gy = CEIL_DIV(num_ty, ty);
    int gx = CEIL_DIV(num_tx, tx);

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
    cm_result_check(pCmDev->CreateKernel(program, "Conv2D_kernel", kernel));

    float* image_host = (float*) malloc(H * W * sizeof(float));
    float* filter_host = (float*) malloc(R * S * sizeof(float));

    // init
    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            image_host[i * W + j] = 0.0;
        }
    }

    for (int i = 0; i < R; ++i)
    {
        for (int j = 0; j < S; ++j)
        {
            filter_host[i * S + j] = 0.0;
        }
    }

    float* result_host = (float*) malloc(H_out * W_out * sizeof(float));
    float* result_golden = (float*) malloc(H_out * W_out * sizeof(float));

    //TODO: write the kernel
    kernel->SetKernelArg(0, 4, &H);
    kernel->SetKernelArg(1, 4, &W);
    kernel->SetKernelArg(2, 4, &R);
    kernel->SetKernelArg(3, 4, &S);

    CmQueue* pCmQueue = nullptr;
    cm_result_check(pCmDev->CreateQueue(pCmQueue));

    CmBuffer* image = nullptr;
    cm_result_check(pCmDev->CreateBuffer(H * W * sizeof(float), image));
    CmBuffer* filter = nullptr;
    cm_result_check(pCmDev->CreateBuffer(R * S * sizeof(float), filter));
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
    
    kernel->SetKernelArg(4, sizeof(SurfaceIndex), image_index);
    kernel->SetKernelArg(5, sizeof(SurfaceIndex), filter_index);
    kernel->SetKernelArg(6, sizeof(SurfaceIndex), result_index);

    CmTask* pKernelArray = nullptr;
    cm_result_check(pCmDev->CreateTask(pKernelArray));
    cm_result_check(pKernelArray->AddKernel(kernel));

    CmThreadGroupSpace* groups = nullptr;
    pCmDev->CreateThreadGroupSpace(gx, gy, tx, ty, groups);

    CmEvent* e = nullptr;
    CM_STATUS s;
    long long kernel_ns = 0;
    long long host_ns = 0;

    for (int it = 0; it < nIter; ++it)
    {
        for (int sr = 0; sr < H_out; sr += KH)
        {
            for (int sc = 0; sc < W_out; sc += KW)
            {
                kernel->SetKernelArg(7, 4, &sr);
                kernel->SetKernelArg(8, 4, &sc);
                long long unsigned time_in_ns = 0;
                std::chrono::nanoseconds start = std::chrono::duration_cast<std::chrono::nanoseconds> (
                    std::chrono::system_clock::now().time_since_epoch());
                cm_result_check(pCmQueue->EnqueueWithGroup(pKernelArray, e, groups));
                for (e->GetStatus(s); s != CM_STATUS_FINISHED; e->GetStatus(s));
                std::chrono::nanoseconds end = std::chrono::duration_cast<std::chrono::nanoseconds> (
                    std::chrono::system_clock::now().time_since_epoch());
                host_ns += (end.count() - start.count());
                cm_result_check(e->GetExecutionTime(time_in_ns));
                kernel_ns += time_in_ns;
            }
        }
    }
    double host_time_cost = host_ns / 1000.0 / nIter;
    double kernel_time_cost = kernel_ns / 1000.0 / nIter;

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
                    tmp += image_host[i * W + j] * filter_host[p * S + q];
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

    return -errs;
}


int main(int argc, char** argv)
{
    RunConv2D(/*H=*/1026, /*W=*/1026, /*R=*/3, /*S=*/3, /*gdx=*/16, /*tdy=*/16);
}
