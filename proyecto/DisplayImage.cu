#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>

using namespace cv;
using namespace std;

void imgToDec(unsigned char *imgBin, unsigned char *imgDec, int cols, int rows){//Cols must be cols x 3
    int pixelByChannel = 0;
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            pixelByChannel = 0;
            for(int i = 7; i >= 0; i--){
                if(imgBin[(row * cols + col) * 8 + i] == 1) pixelByChannel += pow(2,7-i);
                // printf("%d\n",pixelByChannel);
            }
            imgDec[row * cols + col] = pixelByChannel;
        }   
    }
}

void imgToBin(unsigned char *imgDec, unsigned char *imgBin, int cols, int rows){//Cols must be cols x 3
    int pixelByChannel = 0;
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            pixelByChannel = imgDec[row * cols + col];
            for(int i = 7; i >= 0; i--){
                imgBin[(row * cols + col) * 8 + i] = pixelByChannel%2;
                pixelByChannel = (pixelByChannel/2);
            }
        }   
    }
}


void hideImage(unsigned char *secImg, unsigned char *covImg, unsigned char *steImg, int cols, int rows){
    int secBit, covBit;
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            for(int i = 7; i >= 4; i--){
                secBit = secImg[(row * cols + col) * 8 + (i-4)];
                covBit = covImg[(row * cols + col) * 8 + (i-4)];
                steImg[(row * cols + col) * 8 + (i-4)] = covBit;
                steImg[(row * cols + col) * 8 + i] = secBit;
            }
        }   
    }
}

void getSecImg(unsigned char *steImg, unsigned char *secImg, int cols, int rows){
    int secBit;
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            for(int i = 7; i >= 4; i--){
                secBit = steImg[(row * cols + col) * 8 + i];
                secImg[(row * cols + col) * 8 + (i-4)] = secBit;
            }
        }   
    }
}

__global__ void imgToBinGPU(unsigned char *imgDec, unsigned char *imgBin, int cols, int rows){//Cols must be cols x 3
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelByChannel = 0;
    if(row < rows && col < (cols * 8){
        pixelByChannel = imgDec[row * cols + col];
        for(int i = 7; i >= 0; i--){
            imgBin[(row * cols + col) * 8 + i] = pixelByChannel % 2;
            // printf("%d",pixelByChannel % 2);
            pixelByChannel = (pixelByChannel / 2);
        } 
    }
}

int main(int argc, char** argv )
{
    unsigned char *h_secImgRGB, *h_secImgBin, *h_secImgRec;
    unsigned char *h_covImgRGB, *h_covImgBin; 
    unsigned char *h_steImgRGB, *h_steImgBin;

    unsigned char *d_secImgRGB, *d_secImgBin, *d_secImgRec;
    unsigned char *d_covImgRGB, *d_covImgBin; 
    unsigned char *d_steImgRGB, *d_steImgBin;

    cudaError_t err = cudaSuccess;

    double timeCPU, timeGPU;
    
    if ( argc != 3 )
    {
        printf("usage: DisplayImage. <secretImg_Path> <coverImg_Path\n");
        return -1;
    }

    Mat secretImg, coverImg, stegoImg, recovImg;

    secretImg = imread(argv[1], 1);
    coverImg = imread(argv[2], 1);

    if ( !secretImg.data )
    {
        printf("No secretImg data \n");
        return -1;
    }

    printf("cov > %d x %d\nsec > %d x %d\n",coverImg.rows, coverImg.cols, secretImg.rows, secretImg.cols);

    int rows = secretImg.rows;
    int cols = secretImg.cols;
    int colsRGB = cols * secretImg.channels();
    int colsRGB_bin = cols * secretImg.channels() * 8;

    int imgSize = sizeof(unsigned char) * cols * rows * secretImg.channels();
    int imgSizeBin = sizeof(unsigned char) * cols * rows * secretImg.channels() * 8;

    h_secImgRGB = (unsigned char*)malloc(imgSize);
    h_secImgBin = (unsigned char*)malloc(imgSizeBin);
    h_secImgRec = (unsigned char*)malloc(imgSize);

    h_covImgRGB = (unsigned char*)malloc(imgSize);
    h_covImgBin = (unsigned char*)malloc(imgSizeBin);

    h_steImgRGB = (unsigned char*)malloc(imgSize);
    h_steImgBin = (unsigned char*)malloc(imgSizeBin);

    err = cudaMalloc((void**)&d_secImgRGB, imgSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_secImgRGB: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMalloc((void**)&d_secImgBin, imgSizeBin);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_secImgBin: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMalloc((void**)&d_secImgRec, imgSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_secImgRec: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMalloc((void**)&d_covImgRGB, imgSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_covImgRGB: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMalloc((void**)&d_covImgBin, imgSizeBin);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_covImgBin: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMalloc((void**)&d_steImgRGB, imgSize);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_steImgRGB: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMalloc((void**)&d_steImgBin, imgSizeBin);
    if(err != cudaSuccess){ printf(" -cudaMalloc d_steImgBin: %s\n",cudaGetErrorString(err)); return 0;}

    h_secImgRGB = secretImg.data;
    h_covImgRGB = coverImg.data;

    err = cudaMemcpy(d_secImgRGB, h_secImgRGB, imgSize, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy d_secImgRGB < h_secImgRGB: %s\n",cudaGetErrorString(err)); return 0;}

    err = cudaMemcpy(d_covImgRGB, h_covImgRGB, imgSize, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){ printf(" -cudaMemcpy d_covImgRGB < h_covImgRGB: %s\n",cudaGetErrorString(err)); return 0;}

    int threads = 32;
    dim3 blockDim(threads,threads);
	dim3 gridDim(ceil((float)rows/blockDim.x), ceil((float)rows/blockDim.y));

    // imgToBinGPU<<<gridDim, blockDim>>>(d_secImgRGB, d_secImgBin, colsRGB, rows);
    // err = cudaDeviceSynchronize();
    // if(err != cudaSuccess){ printf(" -Kernel call imgToBin(secImg): %s\n",cudaGetErrorString(err)); return 0;}
    
    // err = cudaMemcpy(h_secImgBin, d_secImgBin, imgSizeBin, cudaMemcpyDeviceToHost);
    // if(err != cudaSuccess){ printf(" -cudaMemcpy h_secImgBin < d_secImgBin: %s\n",cudaGetErrorString(err)); return 0;}

    // imgToBinGPU<<<gridDim, blockDim>>>(d_covImgRGB, d_covImgBin, colsRGB, rows);
    // err = cudaDeviceSynchronize();
    // if(err != cudaSuccess){ printf(" -Kernel call imgToBin(covImg): %s\n",cudaGetErrorString(err)); return 0;}
    
    // err = cudaMemcpy(h_covImgBin, d_covImgBin, imgSizeBin, cudaMemcpyDeviceToHost);
    // if(err != cudaSuccess){ printf(" -cudaMemcpy h_covImgBin < d_covImgBin: %s\n",cudaGetErrorString(err)); return 0;}
    
    clock_t startCPU = clock();

    imgToBin(h_secImgRGB, h_secImgBin, colsRGB, rows);
    imgToBin(h_covImgRGB, h_covImgBin, colsRGB, rows);
    hideImage(h_secImgBin, h_covImgBin, h_steImgBin, colsRGB, rows);
    imgToDec(h_steImgBin, h_steImgRGB, colsRGB, rows);
    getSecImg(h_steImgBin, h_secImgBin, colsRGB, rows);
    imgToDec(h_secImgBin, h_secImgRGB, colsRGB, rows);

    timeCPU = ((double)(clock() - startCPU))/CLOCKS_PER_SEC;

    printf("%f\n",timeCPU);

    stegoImg.create(rows, cols, CV_8UC3);
    stegoImg.data = h_steImgRGB;

    recovImg.create(rows, cols, CV_8UC3);
    recovImg.data = h_secImgRGB;

    imwrite("stegoImgOut.jpg", stegoImg);
    imwrite("secretImgRec.jpg", recovImg);

    return 0;
}