#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void int_8bin(int k, int *bin){
    int num = k;
    for(int i = 7; i >= 0; i--)
    {
        bin[i] = num%2;
        num = (num/2);
        //printf("%d - %d\n",num,num%2);
    }
}

int bin_int(int *bin){
    int ans = 0;
    for(int i = 7; i >= 0; i--)
    {
        if(bin[i] == 1) ans += pow(2,7-i);
    }
    return ans;
}

void changeBit(int *bin){
    bin[1] = 1;
}

int main(int argc, char** argv )
{
    unsigned char *imageIn, *imageOut;
    int bin[8] = {0,0,0,0,0,0,0,0};
    int pixel;

    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );
    int rows = image.rows;
    int cols = image.cols;
    int rowEnd = rows * image.channels();
    int colEnd = cols * image.channels();

    int imgSize = sizeof(unsigned char) * cols * rows * image.channels();

    imageIn = (unsigned char*)malloc(imgSize);
    imageOut = (unsigned char*)malloc(imgSize);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    imageIn = image.data;

    // pixel = imageIn[4];
    // int_8bin(pixel, bin);
    //printf("%d => ",pixel);
    
    // for(int i = 0; i < 8; i++)
    // {
    //     printf("%d",bin[i]);
    // }
    // printf(" => %d\n",bin_int(bin));

    printf("Size >> %d x %d\nTotal >> %d x %d\n",rows,cols,rows,colEnd);

    int val;
    
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < colEnd; col++)
        {
            val = imageIn[row * colEnd + col];
            int_8bin(val,bin);
            changeBit(bin);
            imageOut[row * colEnd + col] = bin_int(bin);

        }   
    }
    
    Mat imageAns;
    imageAns.create(rows, cols, CV_8UC3);
    imageAns.data = imageOut;

    imwrite("imageOut.jpg", imageAns);
    imwrite("imageIn.jpg", image);


    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    return 0;
}