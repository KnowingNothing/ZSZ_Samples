/*
 * bmp.cpp
 *
 *  Created on: Apr 30, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;

typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned int   DWORD;
typedef unsigned int   LONG;
#pragma pack(2)//必须得写，否则sizeof得不到正确的结果
typedef struct {
    WORD    bfType;
    DWORD   bfSize;
    WORD    bfReserved1;
    WORD    bfReserved2;
    DWORD   bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    DWORD      biSize;
    LONG       biWidth;
    LONG       biHeight;
    WORD       biPlanes;
    WORD       biBitCount;
    DWORD      biCompression;
    DWORD      biSizeImage;
    LONG       biXPelsPerMeter;
    LONG       biYPelsPerMeter;
    DWORD      biClrUsed;
    DWORD      biClrImportant;
} BITMAPINFOHEADER;

//void saveBitmap(int w, int h,unsigned char *pData,int nDatasize )
bool  CNN::bmp8(const float *data, int width, int height, const char *name)
{
    // Define BMP Size
    const int size = width * height;
    const int colorTablesize=1024;  //灰度图
    // Part.1 Create Bitmap File Header
    BITMAPFILEHEADER fileHeader;

    fileHeader.bfType = 0x4D42;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + size;
    fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize;

    // Part.2 Create Bitmap Info Header
    BITMAPINFOHEADER bitmapHeader = { 0 };

    bitmapHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapHeader.biHeight = height;
    bitmapHeader.biWidth = width;
    bitmapHeader.biPlanes = 1;
    bitmapHeader.biBitCount = 8;
    bitmapHeader.biSizeImage = size;
    bitmapHeader.biCompression = 0; //BI_RGB
    bitmapHeader.biXPelsPerMeter = 0;
    bitmapHeader.biYPelsPerMeter = 0;
    bitmapHeader.biClrUsed = 0;
    bitmapHeader.biClrImportant = 0;

    if (14 != sizeof(BITMAPFILEHEADER) || 40 != sizeof(BITMAPINFOHEADER)) {
    	cout << "invalid bmp file:" << sizeof(BITMAPFILEHEADER) << "," <<  sizeof(BITMAPINFOHEADER) << endl;
    	return false;
    }

    BYTE  *clrs = (BYTE *) new BYTE[colorTablesize];
    for (int x=0; x<colorTablesize; x+=4) {
    	clrs[x] = x/4;  //B
    	clrs[x+1] = x/4;  //G
    	clrs[x+2] = x/4;  //R
    	clrs[x+3] = 255; //a
    }

    BYTE  *bits = (BYTE *) new BYTE[size];
    //BYTE  *bits = (BYTE *) malloc(size);
    if (bits == NULL) {
    	return false;
    }
    for (int i=0; i<height; i++) {
    	for (int j=0; j<width; j++) {
    		int index = i*width + j;
            bits[index] = (BYTE)((data[index] + 1)*255.0);
    	}
    }
    // Write to file
    FILE *output = fopen(name, "wb");
    if (output == NULL) {
        printf("Cannot open file!\n");
        return false;
    } else  {
        fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, output);
        fwrite(&bitmapHeader, sizeof(BITMAPINFOHEADER), 1, output);
        fwrite(clrs, colorTablesize, 1, output);
        fwrite(bits, size, 1, output);
        fclose(output);
    }
    delete[] clrs;
    delete[] bits;
    //free(bits);
    return true;
}

bool CNN::saveMiddlePic(int index)
{
	for (int idx=0; idx< num_map_C1_CNN; idx++) {
		string  strname;
		//strname.Format("%s%d%s%d%s", "tmp/" , index , "_C1_" , idx , ".bmp");
        stringstream str;
        str << "tmp/" << index << "_C1_" << idx << ".bmp";
		if (false == bmp8(&neuron_C1[idx], width_image_C1_CNN, height_image_C1_CNN, str.str().c_str())) {
	    	cout << "C1 failed to new BYTE " << idx;
		}
	}
	for (int idx=0; idx< num_map_S2_CNN; idx++) {
		stringstream str;
		str << "tmp/" << index << "_S2_" << idx << ".bmp";
		if (false == bmp8(&neuron_S2[idx], width_image_S2_CNN, height_image_S2_CNN, str.str().c_str())) {
			cout << "S2 failed to new BYTE " << idx;
		}
	}
	for (int idx=0; idx< num_map_C3_CNN; idx++) {
		stringstream str;
		str << "tmp/" << index << "_C3_" << idx << ".bmp";
		if (false == bmp8(&neuron_C3[idx], width_image_C3_CNN, height_image_C3_CNN, str.str().c_str())) {
			cout << "C3 failed to new BYTE " << idx;
		}
	}
	return true;
}




