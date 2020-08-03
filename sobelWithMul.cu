#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
//#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

//named space for opencv and cout
using namespace std;
using namespace cv;

//kernel function for gpu
/*
* components for the kernel function:
* inout image data
* output image data
* image height
* image width
* transfer matrix in x direction
* transfer matrix in y direction
*/

__global__ void sobelGpu(unsigned char *input, unsigned char *output, int imgH, int imgW, int *d_sobel_x, int *d_sobel_y) {

    //computing with multiple threads
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIndex + yIndex * imgW;

    //gradient in x and y direction
    int Gx = 0;
    int Gy = 0;

    while(offset < (imgH - 2) * (imgW - 2)) {
        //gradient in x direction
        Gx = d_sobel_x[0] * input[(yIndex) * imgW + xIndex] + d_sobel_x[1] * input[(yIndex +1 ) * imgW + xIndex] 
            + d_sobel_x[2] * input[(yIndex + 2) * imgW + xIndex] + d_sobel_x[3] * input[(yIndex) * imgW + xIndex + 1]
            + d_sobel_x[4] * input[(yIndex+1) * imgW + xIndex + 1] + d_sobel_x[5] * input[(yIndex + 2) * imgW + xIndex + 1]
            + d_sobel_x[6] * input[(yIndex) * imgW + xIndex + 2] + d_sobel_x[7] * input[(yIndex+1) * imgW + xIndex + 2]
            + d_sobel_x[8] * input[(yIndex + 2) * imgW + xIndex + 2];

        //gradient in y direction
        Gy = d_sobel_y[0] * input[(yIndex) * imgW + xIndex] + d_sobel_y[1] * input[(yIndex +1 ) * imgW + xIndex]
            + d_sobel_y[2] * input[(yIndex + 2) * imgW + xIndex] + d_sobel_y[3] * input[(yIndex) * imgW + xIndex + 1]
            + d_sobel_y[4] * input[(yIndex+1) * imgW + xIndex + 1] + d_sobel_y[5] * input[(yIndex + 2) * imgW + xIndex + 1]
            + d_sobel_y[6] * input[(yIndex) * imgW + xIndex + 2] + d_sobel_y[7] * input[(yIndex+1) * imgW + xIndex + 2]
            + d_sobel_y[8] * input[(yIndex + 2) * imgW + xIndex + 2];
        
        int sum = abs(Gx) + abs(Gy);
        // constrain the sum with 255
        if (sum > 255) {
            sum = 255;
        }
        output[offset] = sum;
        xIndex += blockDim.x * gridDim.x;
        if(xIndex > imgW - 2) {
            yIndex += blockDim.y * gridDim.y;
            xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        }

        offset = xIndex + yIndex * imgW;
    }
}



// the main function
int main() {
    
    //read the input image, and transfer it in grayscal
    Mat gray_img = imread("test01.jpg", 0);

    // save the gray image
    /*
    save the gray image if needed
    */
    //imwrite("Gray_Image.jpg", gray_img);

    //transfer matrix
    int sobel_x[3][3];
    int sobel_y[3][3];

    //image size, height and width
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

    // assign values to the x direction
    sobel_x[0][0] = -1; sobel_x[0][1] = 0; sobel_x[0][2] =1;
    sobel_x[1][0] = -2; sobel_x[1][1] = 0; sobel_x[1][2] =2;
    sobel_x[2][0] = -1; sobel_x[2][1] = 0; sobel_x[2][2] =1;
    // asign values to the y direction
    sobel_y[0][0] = -1; sobel_y[0][1] = -2; sobel_y[0][2] = -1;
    sobel_y[1][0] = 0; sobel_y[1][1] = 0; sobel_y[1][2] = 0;
    sobel_y[2][0] = 1; sobel_y[2][1] = 2; sobel_y[2][2] = 1;

    //the image for data after processed by GPU
    Mat out_img(imgH, imgW, CV_8UC1, Scalar(0));
    

    /*
    implemetation for GPU kernel
    */

    //device variables for transfer matrixes
    int *d_sobel_x;
    int *d_sobel_y;

    //device memory
    unsigned char *d_in;
    unsigned char *d_out;

    //recording the time
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    //start recording
    cudaEventRecord( start, 0 );

    //memory allocate
    cudaMalloc((void**)&d_in, imgH * imgW * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgH * imgW * sizeof(unsigned char));
    cudaMalloc((void**)&d_sobel_x, 9 * sizeof(int));
    cudaMalloc((void**)&d_sobel_y, 9 * sizeof(int));

    //pass the image data into the GPU
    cudaMemcpy(d_in, gaussImg.data, imgH * imgW * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_sobel_x, (void*)sobel_x, 3 *3* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_sobel_y, (void*)sobel_y, 3 *3* sizeof(int), cudaMemcpyHostToDevice);
    
    //dim3 threadsPerBlock(32, 32);
    //dim3 blocksPerGrid((imgW + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //define the dimentions
    dim3 blocks((int)((imgW+31)/32), (int)(imgH+31)/32);
    dim3 threads(16, 16);

    //call the kernel function
    sobelGpu <<<blocks,threads>>> (d_in, d_out, imgH, imgW, d_sobel_x, d_sobel_y);
    //sobelInCuda3 <<< 1,1 >>> (d_in, d_out, imgH, imgW);

    //pass the output image data back to host
    cudaMemcpy(out_img.data, d_out, imgH * imgW * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //stop recording time
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    //free memory
    cudaFree(d_in);
    cudaFree(d_out);

    //compute the time for execution
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cout << "Time for execution with organized threads and block dimention is: " << static_cast<double>(elapsedTime) << " ms." <<endl;
    //printf( "The time for execution with ognized threads and block dimentions: %.6f ms \n", elapsedTime);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    //save the output image
    imwrite("gpu2.jpg", out_img);

    return 0;
}
