/*
 *
 * Author: Size Zheng
 * Date: 2019-11-02
 * Description: The kernel for simple 2D convolution
 * Notes: Only 3x3 kernel and 4x4 block now
 * 
*/
#define GEM_KERNEL

#undef CM_DEBUG

#include <cm/cm.h>

#define ALIGN(x, y) ((x + y - 1) & ~(y - 1))
#define MAX(x, y) (x > y? x : y)
#define MIN(x, y) (x < y? x : y)
#define SLM_SIZE 256
#define NUM_LANE 4
#define NUM_P 128
#define KERNEL_H 3
#define KERNEL_W 3
#define SLM_ALIGN_LEN 8

// process 4 x 4 box of image for 3 x 3 convolution

extern "C" _GENX_MAIN_ void conv2d_kernel_3x3_b8x4(
    int W,    // width of output data
    int lw,   // width of input data (not original image)
    SurfaceIndex image, SurfaceIndex filter, SurfaceIndex result)   
{
    cm_slm_init(SLM_SIZE);
    unsigned int weights = cm_slm_alloc(SLM_SIZE);
    // load weight data
    cm_slm_load(weights, filter, 0, SLM_SIZE);

    // notice the order of x and y
    unsigned int gx = cm_group_id(1);
    unsigned int gy = cm_group_id(0);
    unsigned int gdx = cm_local_size(1);
    unsigned int gdy = cm_local_size(0);
    unsigned int tx = cm_local_id(1);
    unsigned int ty = cm_local_id(0);

    matrix<float, NUM_P + KERNEL_H - 1, NUM_LANE + KERNEL_W - 1> img_regs;
    matrix<float, NUM_P, NUM_LANE + KERNEL_W - 1> res_regs = 0.0;
    vector<float, ALIGN(KERNEL_H * (NUM_LANE + KERNEL_W - 1), SLM_ALIGN_LEN)> weight_regs;
    vector<unsigned int, ALIGN(KERNEL_H * (NUM_LANE + KERNEL_W - 1), SLM_ALIGN_LEN)> addr = 0;

    // init addr
    // we can actually use const global arrays to initialize addr
    // but that is not general
    // using for loop is general but not good for performance
    // loop from 1 because the first row is already initialized to be zeros
    for (int it = 1; it < ALIGN(KERNEL_H * (NUM_LANE + KERNEL_W - 1), SLM_ALIGN_LEN) / (NUM_LANE + KERNEL_W - 1); ++it)
    {
        addr.select<(NUM_LANE + KERNEL_W - 1), 1>(it * (NUM_LANE + KERNEL_W - 1)) = it * KERNEL_H;
    }
    // disable the redundant address intorduced by alignment, not necessary
    // addr.select<ALIGN(KERNEL_H * (NUM_LANE + KERNEL_W - 1), SLM_ALIGN_LEN) - KERNEL_H * (NUM_LANE + KERNEL_W - 1), 1>(KERNEL_H * (NUM_LANE + KERNEL_W - 1)) = 0;

    // load image data row by row because our dataport is general 1D surface, they should be aligned
    // so the rows are not adjacent and can't be loaded at a time
    // however, different rows can be loaded simultaneously, which should be implemented later
    #pragma unroll
    for (int i = 0; i < NUM_P + KERNEL_H - 1; ++i)
    {
        read(image, (((gx * gdx + tx) * NUM_P + i) * lw + (gy * gdy + ty) * NUM_LANE) * sizeof(float), img_regs.row(i));
    }

    // compute
    // this imitates the SC'19 paper
    #pragma unroll
    for (int i = 0; i < NUM_P; ++i)
    {
        #pragma unroll
        for (int m = 0; m < KERNEL_W; ++m)
        {
            // shuffle
            // shuffle becomes simply shift the register vectors
            res_regs.select<1, 1, NUM_LANE + KERNEL_W - 2, 1>(i, 1) = res_regs.select<1, 1, NUM_LANE + KERNEL_W - 2, 1>(i, 0);
            matrix<float, KERNEL_H, NUM_LANE + KERNEL_W - 1> tmp = 0.0;
            // load weights to registers
            // the SLM limits us to load only 8/16 elements at a time
            for (int it = 0; it < ALIGN(KERNEL_H * (NUM_LANE + KERNEL_W - 1), SLM_ALIGN_LEN) / SLM_ALIGN_LEN; ++it)
            {
                cm_slm_read(weights, addr.select<SLM_ALIGN_LEN, 1>(it * SLM_ALIGN_LEN), weight_regs.select<SLM_ALIGN_LEN, 1>(it * SLM_ALIGN_LEN));
            }            
            tmp = img_regs.select<KERNEL_H, 1, NUM_LANE + KERNEL_W - 1, 1>(i, 0) * weight_regs.select<KERNEL_H * (NUM_LANE + KERNEL_W - 1), 1>(0);
            #pragma unroll
            for (int n = 0; n < KERNEL_H; ++n)
            {
                res_regs.select<1, 1, NUM_LANE + KERNEL_W - 1, 1>(i, 0) += tmp.select<1, 1, NUM_LANE + KERNEL_W - 1, 1>(n, 0);
            }
            // update address
            addr += 1;
            // for (int kkk = 0; kkk < NUM_P; ++kkk)
            // {
            //     res_regs.row(i) = img_regs.row(i);
            // }
        }
        // restore address for next row of points
        addr -= KERNEL_W;
    }

    // write out results, is naturally aligned because we use 4x4 block
    for (int i = 0; i < NUM_P; ++i)
    {
        write(
            result, 
            (((gx * gdx + tx) * NUM_P + i) * W + (gy * gdy + ty) * NUM_LANE) * sizeof(float), 
            res_regs.row(i).select<NUM_LANE, 1>(KERNEL_W - 1));
    }
    
}


#define M 8
#define N 16

extern "C" _GENX_MAIN_ void conv2d_kernel_3x3(SurfaceIndex img, SurfaceIndex filt, SurfaceIndex res)
{
    matrix<float, M + 2, N + 2> in = 3.0f;
    matrix<float, 3, 3> w;
    matrix<float, M, N> out = 0.0f;

    uint h_pos = get_thread_origin_x();
    uint v_pos = get_thread_origin_y();

    #pragma unroll
    for (int i = 0; i < (M + 2) / 8; ++i)
    {
        #pragma unroll
        for (int j = 0; j < (N + 2) / 8; ++j)
        {
            read(img, h_pos * N * 4 + j * 8 * 4, v_pos * M + i * 8, in.select<8, 1, 8, 1>(i * 8, j * 8));
        }
        if ((N + 2) / 8 * 8 < N + 2)
        {
            read(img, h_pos * N * 4 + (N + 2) / 8 * 8 * 4, v_pos * M + i * 8, in.select<8, 1, N + 2 - (N + 2) / 8 * 8, 1>(i * 8, (N + 2) / 8 * 8));
        }
    }
    if ((M + 2) / 8 * 8 < M + 2)
    {
        #pragma unroll
        for (int j = 0; j < (N + 2) / 8; ++j)
        {
            read(img, h_pos * N * 4 + j * 8 * 4, v_pos * M + (M + 2) / 8 * 8, in.select<M + 2 - (M + 2) / 8 * 8, 1, 8, 1>((M + 2) / 8 * 8, j * 8));
        }
        if ((N + 2) / 8 * 8 < N + 2)
        {
            read(img, h_pos * N * 4 + (N + 2) / 8 * 8 * 4, v_pos * M + (M + 2) / 8 * 8, in.select<M + 2 - (M + 2) / 8 * 8, 1, N + 2 - (N + 2) / 8 * 8, 1>((M + 2) / 8 * 8, (N + 2) / 8 * 8));
        }
    }
    //read(img, h_pos * N * 4, v_pos * M, in);
    read(filt, 0, 0, w);

    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            out += w.replicate<M * N, 1>(i, j) * in.select<M, 1, N, 1>(i, j);
        }
    }   

    #pragma unroll
    for (int i = 0; i < M / 8; ++i)
    {
        #pragma unroll
        for (int j = 0; j < N / 8; ++j)
        {
            write(res, h_pos * N * 4 + j * 8 * 4, v_pos * M + i * 8, out.select<8, 1, 8, 1>(i * 8, j * 8));
        }
        // if (N / 8 * 8 < N)
        // {
        //     write(res, h_pos * N * 4 + N / 8 * 8 * 4, v_pos * M + i * 8, out.select<8, 1, N - N / 8 * 8, 1>(i * 8, N / 8 * 8));
        // }
    }
    // if (M / 8 * 8 < M)
    // {
    //     for (int j = 0; j < N / 8; ++j)
    //     {
    //         write(res, h_pos * N * 4 + j * 8 * 4, v_pos * M + M / 8 * 8, out.select<M - M / 8 * 8, 1, 8, 1>(M / 8 * 8, j * 8));
    //     }
    //     if (N / 8 * 8 < N)
    //     {
    //         write(res, h_pos * N * 4 + N / 8 * 8 * 4, v_pos * M + M / 8 * 8, out.select<M - M / 8 * 8, 1, N - N / 8 * 8, 1>(M / 8 * 8, N / 8 * 8));
    //     }
    // }
    //write(res, h_pos * N * 4, v_pos * M, out);
}