#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>

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
                // printf("%d => %d at %d\n",pixelByChannel, imgBin[(row * cols + col) * 8 + i],((row * cols + col) * 8 + i));
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

int main(int argc, char** argv )
{
    unsigned char *secretImgDec, *secretImgBin, *secretImgOut;
    unsigned char *coverImgDec, *coverImgBin; 
    unsigned char *stegoImgDec, *stegoImgBin;

    // unsigned char matTest[6] = {213,167,130,145,125,118};
    // unsigned char matBin[48];
    // unsigned char matOut[6];

    // matTest = (unsigned char*)malloc(sizeof(unsigned char) * 2 * 2);
    // matBin = (unsigned char*)malloc(sizeof(unsigned char) * 2 * 2 * 3 * 8);
    
    if ( argc != 3 )
    {
        printf("usage: DisplayImage. <secretImg_Path> <coverImg_Path\n");
        return -1;
    }

    Mat secretImg, coverImg, stegoImg, recovImg;

    secretImg = imread(argv[1], 1);
    coverImg = imread(argv[2], 1);

    printf("cov > %d x %d\nsec > %d x %d\n",coverImg.rows, coverImg.cols, secretImg.rows, secretImg.cols);

    int rows = secretImg.rows;
    int cols = secretImg.cols;
    int colsRGB = cols * secretImg.channels();
    int colsRGB_bin = cols * secretImg.channels() * 8;

    int imgSize = sizeof(unsigned char) * cols * rows * secretImg.channels();
    int imgSizeBin = sizeof(unsigned char) * cols * rows * secretImg.channels() * 8;

    secretImgDec = (unsigned char*)malloc(imgSize);
    secretImgOut = (unsigned char*)malloc(imgSize);
    secretImgBin = (unsigned char*)malloc(imgSizeBin);

    coverImgDec = (unsigned char*)malloc(imgSize);
    coverImgBin = (unsigned char*)malloc(imgSizeBin);

    stegoImgDec = (unsigned char*)malloc(imgSize);
    stegoImgBin = (unsigned char*)malloc(imgSizeBin);

    if ( !secretImg.data )
    {
        printf("No secretImg data \n");
        return -1;
    }

    secretImgDec = secretImg.data;
    coverImgDec = coverImg.data;

    imgToBin(secretImgDec, secretImgBin, colsRGB, rows);
    imgToBin(coverImgDec, coverImgBin, colsRGB, rows);
    hideImage(secretImgBin, coverImgBin, stegoImgBin, colsRGB, rows);
    imgToDec(stegoImgBin, stegoImgDec, colsRGB, rows);
    getSecImg(stegoImgBin, secretImgBin, colsRGB, rows);
    imgToDec(secretImgBin, secretImgDec, colsRGB, rows);

    // imgToBin(matTest, matBin, 3, 2);
    // imgToDec(matBin, matOut, 3, 2);

    // printf("Size >> %d x %d\nTotal >> %d x %d\n",rows,cols,rows,colsRGB);

    // for(int row = 0; row < 2; row++)
    // {
    //     for(int col = 0; col < 24; col++)
    //     {
    //         int bin = matBin[row * 24 + col];
    //         printf("%d",bin);
    //     }
    //     printf("\n");
    // }

    // for(int row = 0; row < 2; row++)
    // {
    //     for(int col = 0; col < 3; col++)
    //     {
    //         int bin = matOut[row * 3 + col];
    //         printf("%d,",bin);
    //     }
    //     printf("\n");
    // }
    
    
    // for(int row = 0; row < rows; row++)
    // {
    //     for(int col = 0; col < colsRGB_bin; col++)
    //     {
    //         int bin = secretImgBin[row * colsRGB_bin + col];
    //         printf("%d",bin);

    //     }   
    // }
    
    stegoImg.create(rows, cols, CV_8UC3);
    stegoImg.data = stegoImgDec;

    recovImg.create(rows, cols, CV_8UC3);
    recovImg.data = secretImgDec;

    imwrite("stegoImgOut.jpg", stegoImg);
    imwrite("secretImgOut.jpg", secretImg);


    // namedWindow("Display secretImg", WINDOW_AUTOSIZE );
    // imshow("Display secretImg", secretImg);

    return 0;
}