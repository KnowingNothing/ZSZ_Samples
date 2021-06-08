#ifndef __rgb2gray_h__
#define __rgb2gray_h__

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"

#include "imgproc/xf_cvt_color.hpp"

#define IMG_MAX_ROWS 1080
#define IMG_MAX_COLS 1920

void rgb2gray(hls::stream<ap_axiu<24,1,1,1>>&video_in,hls::stream<ap_axiu<24,1,1,1>>&video_out);

#endif 
