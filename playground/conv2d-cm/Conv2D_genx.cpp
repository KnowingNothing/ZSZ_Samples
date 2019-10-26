#define GEM_KERNEL

#undef CM_DEBUG

#include <cm/cm.h>


// process 16 x 16 box of image

extern "C" _GENX_MAIN_ void conv2d_kernel(int H, int W, int R, int S, SurfaceIndex image, SurfaceIndex filter, SurfaceIndex result, int sr, int sc)
{

}