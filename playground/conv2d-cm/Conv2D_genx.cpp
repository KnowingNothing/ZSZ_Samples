#define GEM_KERNEL

#undef CM_DEBUG

#include <cm/cm.h>


// process 16 x 16 box of image

extern "C" _GENX_MAIN_ void conv2d_kernel(
    int H, int W, int R, int S, 
    SurfaceIndex image, SurfaceIndex filter, SurfaceIndex result, 
    int sr, int sc)
{
    int H_out = (H - R) + 1;
    int W_out = (W - S) + 1;
    // return 0 to check host program
    vector<float, 16 * 16> v = 0;
    int gid = cm_linear_global_id();
    write(result, gid * sizeof(float), v);
}