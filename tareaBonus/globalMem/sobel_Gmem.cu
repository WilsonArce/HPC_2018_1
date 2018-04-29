#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ void gpuGrayScale(unsigned char *imgIn, unsigned char *imgOut, int cols, int rows){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned char r,g,b;
  if((row < rows) && (col < cols)){
    r = imgIn[(row * cols + col) * 3 + 2];
    g = imgIn[(row * cols + col) * 3 + 1];
    b = imgIn[(row * cols + col) * 3 + 0];

    imgOut[row * cols + col] = r * 0.299 + g * 0.587 + b * 0.114;
  }
}

__global__ void gpuSobelFilter(unsigned char *imgGray, unsigned char *imgFiltered, \
  unsigned char *imgX, unsigned char *imgY, int cols, int rows){
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.y + threadIdx.x;

  int xFilter[9] = {-1,0,1,-2,0,2,-1,0,1};
	int yFilter[9] = {-1,-2,-1,0,0,0,1,2,1};

  int sbCols, sbRows, sumx, sumy, x, y, ci, cj;
  sbCols = sbRows = 3;

  //for(i = 0; i < rows; i++){
	//	for(j = 0; j < cols; j++){
    if((i < rows) && (j < cols)){
			sumx = 0; sumy = 0; ci = i-2;
			for(y = 0; y < sbRows; y++){
				ci++;
				cj = j-1;
				for(x = 0; x < sbCols; x++){
					if(ci < 0 || cj < 0){
						sumx += 0;
						sumy += 0;
					}else{
						sumx += imgGray[ci * cols + cj] * xFilter[y * sbCols + x];
						sumy += imgGray[ci * cols + cj] * yFilter[y * sbCols + x];
					}
					cj++;
				}
			}
			if(sumx > 255){
				imgX[i * cols + j] = 255;
			}else{
				if(sumx < 0){
					imgX[i * cols + j] = 0;
				}else{
					imgX[i * cols + j] = sumx;
				}
			}
			if(sumy > 255){
				imgY[i * cols + j] = 255;
			}else{
				if(sumy < 0){
					imgY[i * cols + j] = 0;
				}else{
					imgY[i * cols + j] = sumy;
				}
			}
			imgFiltered[i * cols + j] = sqrt(powf(imgX[i * cols + j],2) + powf(imgY[i * cols + j],2));
		}
	//}

}


int main(int argc, char** argv )
{

  double timeGPU_GS, timeGPU_SB, timeCPU_GS, timeCPU_SB;

  //elements for GRAYSCALE filter
  unsigned char *h_imageIn, *h_imageGray, *d_imageIn, *d_imageGray;

  //elements for SOBEL filter
  unsigned char *h_imageSobel, *d_imageX, *d_imageY, *d_imageSobel;

  //char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  //cudaError_t error = cudaSuccess;
  Mat image;
  image = imread( argv[1], 1 );
  
  if ( argc != 2 )
  {
    printf("usage: DisplayImage <Image_Path>\n");
    return -1;
  }

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ///Element for openCV transformations
  Mat src_gray;
  Mat grad;

  clock_t startCPU_GS = clock();
  /// Convert it to gray
  cvtColor( image, src_gray, CV_BGR2GRAY );
  timeCPU_GS = ((double)(clock() - startCPU_GS))/CLOCKS_PER_SEC;

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  clock_t startCPU_SB = clock();
  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  timeCPU_SB = ((double)(clock() - startCPU_SB))/CLOCKS_PER_SEC;
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  //image size cols and rows
  int cols = image.cols;
  int rows = image.rows;

  //size of initial image and result images
  int imgInSize = sizeof(unsigned char) * cols * rows * image.channels();
  int imgOutSize = sizeof(unsigned char) * cols * rows;

  //allocation of memory for elements of GRAYSCALE filter ON HOST
  h_imageIn = (unsigned char*)malloc(imgInSize);
  h_imageGray = (unsigned char*)malloc(imgOutSize);

  //allocation of memory for elements of GRAYSCALE filter ON DEVICE
  cudaMalloc((void**)&d_imageIn, imgInSize);
  cudaMalloc((void**)&d_imageGray, imgOutSize);

  //allocation of memory for elements of SOBEL filter ON HOST
  h_imageSobel = (unsigned char*)malloc(imgOutSize);

  //allocation of memory for elements of SOBEL filter ON DEVICE
  cudaMalloc((void**)&d_imageX, imgOutSize);
  cudaMalloc((void**)&d_imageY, imgOutSize);
  cudaMalloc((void**)&d_imageSobel, imgOutSize);

  //error = cudaMalloc((void**)&d_imageIn, imgInSize);
  /*if(error != cudaSuccess){
      printf("Error reservando memoria para d_imageIn\n -> %s\n", cudaGetErrorString(error));
      exit(-1);
  }*/
  
  //passing data for image processing
  h_imageIn = image.data;

  //passing image data from HOST to DEVICE for GRAYSCALE filter
  cudaMemcpy(d_imageIn, h_imageIn, imgInSize, cudaMemcpyHostToDevice);

  //parameters definition for CUDA kernel
  int threads = 32;
  dim3 numThreads(threads, threads);
  dim3 blockDim(ceil(cols/float(threads)), ceil(rows/float(threads)));

  clock_t startGPU_GS = clock();
  //CUDA grayscale kernel call
  gpuGrayScale<<<blockDim, numThreads>>>(d_imageIn, d_imageGray, cols, rows);
  cudaDeviceSynchronize();//CUDA threads sincronization
  timeGPU_GS = ((double)(clock() - startGPU_GS))/CLOCKS_PER_SEC;

  //passing result GRAYSCALE data from DEVICE to HOST
  cudaMemcpy(h_imageGray, d_imageGray, imgOutSize, cudaMemcpyDeviceToHost);

  clock_t startGPU_SB = clock();
  //CUDA sobel filter call
  gpuSobelFilter<<<blockDim, numThreads>>>(d_imageGray, d_imageSobel, d_imageX, d_imageY, cols, rows);
  cudaDeviceSynchronize();//CUDA threads sincronization
  timeGPU_SB = ((double)(clock() - startGPU_SB))/CLOCKS_PER_SEC;

  //passing result SOBEL data from DEVICE to HOST
  cudaMemcpy(h_imageSobel, d_imageSobel, imgOutSize, cudaMemcpyDeviceToHost);

  Mat imageOut;
  imageOut.create(rows, cols, CV_8UC1);
  imageOut.data = h_imageSobel;

  //printf("**Global memory implementation**\n");
  //cout<<"Image size = "<< image.size() << endl;
  //printf("Global memory\nGrayscale time\n");
  printf("%f",timeGPU_SB);
  //printf("Sobel filter time\n");
  //printf("sf{%f-%f}",timeCPU_SB,timeGPU_SB);
  //printf("  CPU = %f s\n",timeCPU_SB);
  //printf("  GPU = %f s\n",timeGPU_SB);

  imwrite("imageSobel_gpu.jpg", imageOut);
  imwrite("imageSobel_opCV.jpg", grad);

  //waitKey(0);

  //memory deallocation on DEVICE
  cudaFree(d_imageIn);
  cudaFree(d_imageGray);
  cudaFree(d_imageX);
  cudaFree(d_imageY);

  return 0;
}