#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
  if ( argc != 2 )
  {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }

  Mat image;
  image = imread( argv[1], 1 );

  if ( !image.data )
  {
    printf("No image data \n");
    return -1;
  }
  //namedWindow("Display Image", WINDOW_AUTOSIZE );
  //imshow("Display Image", image);

  Mat img = image;
  float r,g,b;
  for(int y=0;y<image.rows;y++){
    for(int x=0;x<image.cols;x++){
      // get pixel
      Vec3b color = img.at<Vec3b>(Point(x,y));

      r = color[0];
      g = color[1];
      b = color[2];

      //I = .299f * R + .587f * G + .114f * B
      color[2] = (r * 0.299 + g * 0.587 + b * 0.114);
      color[1] = (r * 0.299 + g * 0.587 + b * 0.114);
      color[0] = (r * 0.299 + g * 0.587 + b * 0.114);

      // set pixel
      img.at<Vec3b>(Point(x,y)) = color;
    }
  }

  imwrite("lena_out.jpg", img);

  //cout << image << endl;

  //waitKey(0);

  return 0;
}
