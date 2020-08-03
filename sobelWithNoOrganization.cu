#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
//#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>  
#include <iostream>


//namespace for cout and opencv
using namespace std;
using namespace cv;


//kernel function for gpu
/*
* components for the kernel function:
* input image data
* output image data
* image height
* image width
*/

__global__ void sobelGpu2(unsigned char *input, unsigned char *output, int imgH, int imgW) {
    
    //computing with 1 thread/block and 1 block/grid
    int xIndex = threadIdx.x;
    int yIndex = xIndex/imgW;
    int index = 0;

    //gradient in x and y direction
    int Gx = 0;
    int Gy = 0;

    while(yIndex < (imgH - 2)) {
        
        //gradient in x direction
        Gx = (-1)  * input[(yIndex) * imgW + xIndex] + 0*input[(yIndex +1 ) * imgW + xIndex]
            + 1*input[(yIndex + 2) * imgW + xIndex] + (-2) * input[(yIndex) * imgW + xIndex + 1]
            + 0 * input[(yIndex+1) * imgW + xIndex + 1] + 2 * input[(yIndex + 2) * imgW + xIndex + 1]
            + (-1) * input[(yIndex) * imgW + xIndex + 2] + 0 * input[(yIndex+1) * imgW + xIndex + 2]
            + 1 * input[(yIndex + 2) * imgW + xIndex + 2];
        //gradient in y direction
        Gy = (-1) * input[(yIndex) * imgW + xIndex] + (-2) * input[(yIndex +1 ) * imgW + xIndex]
            + (-1)*input[(yIndex + 2) * imgW + xIndex] + (0) * input[(yIndex) * imgW + xIndex + 1]
            + 0 * input[(yIndex+1) * imgW + xIndex + 1] + 0 * input[(yIndex + 2) * imgW + xIndex + 1]
            + (1) * input[(yIndex) * imgW + xIndex + 2] + 2 * input[(yIndex+1) * imgW + xIndex + 2]
            + 1 * input[(yIndex + 2) * imgW + xIndex + 2];
        
        int sum = abs(Gx) + abs(Gy);
        //constrain
        if (sum > 255) {
            sum = 255; 
        }
        output[index] = sum;
        xIndex ++;
        if ( xIndex == imgW-2 ) {
            xIndex = 0;
            yIndex ++;
        }

        index = xIndex + yIndex * imgW;
    }
}

//the main execution
int main() {

    //read the input image, and transfer it in grayscal
    Mat gray_img = imread("test01.jpg", 0);


    // save the gray image
    /*
    save the gray image if needed
    */
    //imwrite("Gray_Image.jpg", gray_img);

    //image size
    int imgH = gray_img.rows;
    int imgW = gray_img.cols;

    //initialze the image after gauss filter
    Mat gaussImg;
    //implementation of the gauss filter with a 3 X 3 kernel
    GaussianBlur(gray_img, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);
    // save the gauss image
    /*
    save the image after gauss filter if needed
    */
    //imwrite("gauss.jpg", gaussImg);

    //the image for data after processed by GPU
    Mat out_img(imgH, imgW, CV_8UC1, Scalar(0));


    /*
    implemetation for GPU kernel
    */

    //device variables for transfer matrixes
    //device memory
    unsigned char *d_in;
    unsigned char *d_out;

    //record the time
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    //start recording
    cudaEventRecord( start, 0 );

    //memory allocate
    cudaMalloc((void**)&d_in, imgH * imgW * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgH * imgW * sizeof(unsigned char));

    //pass the image data into the GPU
    cudaMemcpy(d_in, gaussImg.data, imgH * imgW * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //define the dimentions
    //dim3 threadsPerBlock(32, 32);
    //dim3 blocksPerGrid((imgW + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //call the kernel function
    sobelGpu2 <<< 1,1 >>> (d_in, d_out, imgH, imgW);

    //pass the output image data back to host
    cudaMemcpy(out_img.data, d_out, imgH * imgW * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //stop recording
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    //free memory
    cudaFree(d_in);
    cudaFree(d_out);

    //compute the time for execution
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cout << "Time for execution with 1 thread/block and 1 block/grid is : " << static_cast<double>(elapsedTime) << " ms." << endl;
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    //save the output image
    imwrite("gpu1.jpg", out_img);

    return 0;
}

