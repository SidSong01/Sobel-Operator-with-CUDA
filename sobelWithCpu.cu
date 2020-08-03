#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
//#include <cstdio>


//name space for cout and opencv
using namespace std;
using namespace cv;

//global variables
Mat sobel_x, sobel_y; 
Mat abs_x, abs_y;
//for the out/input image data and gauss filter
Mat pack_img, gauss_img, grayimg, output;
//size of images
int imgH, imgW;

//preprocessing for the image data
void pre_process(void) {
    //read the image in grayscale
    grayimg = imread("test01.jpg", 0);
    //save the image
    imwrite("grayscale.jpg", grayimg);

    imgH = grayimg.rows;
    imgW = grayimg.cols;

    //implementing gaussian filter
    GaussianBlur(grayimg, gauss_img, Size(3, 3), 0, 0, BORDER_DEFAULT);
    //save the image
    imwrite("gaussian_filter.jpg", gauss_img);
    //output = gauss_img;
}

//sobel in CPU with processing on pixels
void sobelCpu() {
    
    //for recording time
    clock_t begin, end;
    //start recording
    begin = clock();
    //transfer matrix
    int sobel_x[3][3];
    int sobel_y[3][3];
    //assign values to the x direction
    sobel_x[0][0] = -1; sobel_x[0][1] = 0; sobel_x[0][2] =1;
    sobel_x[1][0] = -2; sobel_x[1][1] = 0; sobel_x[1][2] =2;
    sobel_x[2][0] = -1; sobel_x[2][1] = 0; sobel_x[2][2] =1;
    //assign values to the y direction
    sobel_y[0][0] = -1; sobel_y[0][1] = -2; sobel_y[0][2] = -1;
    sobel_y[1][0] = 0; sobel_y[1][1] = 0; sobel_y[1][2] = 0;
    sobel_y[2][0] = 1; sobel_y[2][1] = 2; sobel_y[2][2] = 1;

    Mat img = gauss_img;
    //Mat newimg = img;

    for (int j = 0; j<img.rows-2; j++) {
       for (int i = 0; i<img.cols-2; i++) {

           //computing the sobel gradient in x and y direction
           int g_x = (sobel_x[0][0] * (int)img.at<uchar>(j,i)) + (sobel_x[0][1] * (int)img.at<uchar>(j+1,i)) 
               + (sobel_x[0][2] * (int)img.at<uchar>(j+2,i)) + (sobel_x[1][0] * (int)img.at<uchar>(j,i+1))
               + (sobel_x[1][1] * (int)img.at<uchar>(j+1,i+1)) + (sobel_x[1][2] * (int)img.at<uchar>(j+2,i+1)) 
               + (sobel_x[2][0] * (int)img.at<uchar>(j,i+2)) + (sobel_x[2][1] * (int)img.at<uchar>(j+1,i+2))
               + (sobel_x[2][2] * (int)img.at<uchar>(j+2,i+2));
         
           int g_y = (sobel_y[0][0] * (int)img.at<uchar>(j,i)) + (sobel_y[0][1] * (int)img.at<uchar>(j+1,i))
               + (sobel_y[0][2] * (int)img.at<uchar>(j+2,i)) + (sobel_y[1][0] * (int)img.at<uchar>(j,i+1))
               + (sobel_y[1][1] * (int)img.at<uchar>(j+1,i+1)) + (sobel_y[1][2] * (int)img.at<uchar>(j+2,i+1))
               + (sobel_y[2][0] * (int)img.at<uchar>(j,i+2)) + (sobel_y[2][1] * (int)img.at<uchar>(j+1,i+2))
               + (sobel_y[2][2] * (int)img.at<uchar>(j+2,i+2));
         
           int sum = abs(g_x) + abs(g_y);
           if (sum > 255) {
               sum = 255;
            }
             img.at<unsigned char>(j,i) = sum;
     }
    }

    //end time
    end = clock();
    cout << "Execution time using CPU operating on pixles: " << static_cast<double>(end - begin) / CLOCKS_PER_SEC*1000 << " ms." << endl;
    //save the output
    imwrite("cpu.jpg", img);
}

//using OpenCV package for sobel
void sobelOpenCV(int, void*) {
    clock_t begin, end;
    begin = clock();

    //gradient in x direnction
    Sobel(grayimg, sobel_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs( sobel_x, abs_x);

    //gradient in y direnction
    Sobel(grayimg, sobel_y, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs( sobel_y, abs_y);

    //combine the gradients
    addWeighted(abs_x, 0.5, abs_y, 0.5,0,pack_img );
    end = clock();
    cout << "Execution time using OpenCV package: " << static_cast<double>(end - begin) / CLOCKS_PER_SEC*1000 << " ms." << endl;
    //save the image
    imwrite("opencv.jpg",pack_img);
}


int main(int argc, char *argv[]) {
    
    //preprocessing
    pre_process();
    // using opencv package
    sobelOpenCV(0,0);
    //cpu operation on pixels
    sobelCpu();
    
    return 0;
}


