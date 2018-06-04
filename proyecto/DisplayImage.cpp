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


void changeBit(int *bin){
    bin[1] = 1;
}

int main(int argc, char** argv )
{
    unsigned char *secretImgDec, *secretImgBin, *coverImgBin, *secretImgOut;
    unsigned char matTest[6] = {213,167,130,145,125,118};
    unsigned char matBin[48];
    unsigned char matOut[6];

    // matTest = (unsigned char*)malloc(sizeof(unsigned char) * 2 * 2);
    // matBin = (unsigned char*)malloc(sizeof(unsigned char) * 2 * 2 * 3 * 8);
    
    if ( argc != 2 )
    {
        printf("usage: DisplaysecretImg.out <secretImg_Path>\n");
        return -1;
    }

    Mat secretImg, coverImg, stegoImg;
    secretImg = imread( argv[1], 1 );
    int rows = secretImg.rows;
    int cols = secretImg.cols;
    int colsRGB = cols * secretImg.channels();
    int colsRGB_bin = cols * secretImg.channels() * 8;

    int imgSize = sizeof(unsigned char) * cols * rows * secretImg.channels();
    int imgSizeBin = sizeof(unsigned char) * cols * rows * secretImg.channels() * 8;

    secretImgDec = (unsigned char*)malloc(imgSize);
    secretImgOut = (unsigned char*)malloc(imgSize);
    secretImgBin = (unsigned char*)malloc(imgSizeBin);

    if ( !secretImg.data )
    {
        printf("No secretImg data \n");
        return -1;
    }

    secretImgDec = secretImg.data;

    imgToBin(secretImgDec, secretImgBin, colsRGB, rows);
    imgToDec(secretImgBin, secretImgOut, colsRGB, rows);
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
    
    Mat secretImgAns;
    secretImgAns.create(rows, cols, CV_8UC3);
    secretImgAns.data = secretImgOut;

    imwrite("secretImgOut.jpg", secretImgAns);
    imwrite("secretImgDec.jpg", secretImg);


    // namedWindow("Display secretImg", WINDOW_AUTOSIZE );
    // imshow("Display secretImg", secretImg);

    return 0;
}