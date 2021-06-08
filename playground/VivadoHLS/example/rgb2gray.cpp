/*
 * @Author: your name
 * @Date: 2021-06-07 17:02:51
 * @LastEditTime: 2021-06-07 17:02:53
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \undefinedc:\Users\mprc\Desktop\rgb2gray.cpp
 */
/*
 * @Author: your name
 * @Date: 2021-06-07 17:02:51
 * @LastEditTime: 2021-06-07 17:02:52
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: \undefinedc:\Users\mprc\Desktop\rgb2gray.cpp
 */
#include "rgb2gray.h"

void ExtractPixel(XF_TNAME(XF_8UC3,XF_NPPC1)&src,ap_uint<8>dst[3])
{
	unsigned int i,j=0;
	for(i=0;i<24;i+=8)
	{
		dst[j]=src.range(i+7,i);
		j++;
	}
}

template<int ROWS,int COLS>
void xfrgb2gray(xf::cv::Mat<XF_8UC3,ROWS,COLS,XF_NPPC1>&src,xf::cv::Mat<XF_8UC1,ROWS,COLS,XF_NPPC1>&dst)
{
	XF_TNAME(XF_8UC3,XF_NPPC1)rgb_packed;
	XF_TNAME(XF_8UC1,XF_NPPC1)gray_packed;
	ap_uint<8>rgb[3];
	ap_uint<8>gray;
	unsigned int i,j=0;
	for(i=0;i<ROWS;i++)
	{
		for(j=0;j<COLS;j++)
		{
			rgb_packed=src.read(i*COLS+j);
			ExtractPixel(rgb_packed,rgb);
			gray=CalculateGRAY(rgb[0],rgb[1],rgb[2]);
			gray_packed.range(7,0)=gray;
			dst.write(i*COLS+j,gray_packed);
		}
	}
}

template<int ROWS,int COLS>
void xfgray2rgb(xf::cv::Mat<XF_8UC1,ROWS,COLS,XF_NPPC1>&src,xf::cv::Mat<XF_8UC3,ROWS,COLS,XF_NPPC1>&dst)
{
	XF_TNAME(XF_8UC1,XF_NPPC1)gray_packed;
	XF_TNAME(XF_8UC3,XF_NPPC1)rgb_packed;
	ap_uint<8>gray;
	unsigned int i,j=0;
	for(i=0;i<ROWS;i++)
	{
		for(j=0;j<COLS;j++)
		{
			gray_packed=src.read(i*COLS+j);
			gray=gray_packed.range(7,0);
			rgb_packed.range(7,0)=gray;
			rgb_packed.range(15,8)=gray;
			rgb_packed.range(23,16)=gray;
			dst.write(i*COLS+j,rgb_packed);
		}
	}
}

void rgb2gray(hls::stream<ap_axiu<24,1,1,1>>&video_in,hls::stream<ap_axiu<24,1,1,1>>&video_out)
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW
#pragma HLS INTERFACE axis port=video_out register_mode=both register
#pragma HLS INTERFACE axis port=video_in register_mode=both register
	xf::cv::Mat<XF_8UC3,IMG_MAX_ROWS,IMG_MAX_COLS,XF_NPPC1>img_rgb_src;
//#pragma HLS stream variable=img_rgb_src.data depth=int(1920) dim=1
	xf::cv::Mat<XF_8UC1,IMG_MAX_ROWS,IMG_MAX_COLS,XF_NPPC1>img_gray_src;
//#pragma HLS stream variable=img_gray_src.data depth=1920 dim=1
	xf::cv::Mat<XF_8UC3,IMG_MAX_ROWS,IMG_MAX_COLS,XF_NPPC1>img_gray_dst;
//#pragma HLS stream variable=img_gray_dst.data depth=1920 dim=1
	xf::cv::AXIvideo2xfMat(video_in,img_rgb_src);
	xfrgb2gray<IMG_MAX_ROWS,IMG_MAX_COLS>(img_rgb_src,img_gray_src);
	xfgray2rgb<IMG_MAX_ROWS,IMG_MAX_COLS>(img_gray_src,img_gray_dst);
	xf::cv::xfMat2AXIvideo(img_gray_dst,video_out);
}
