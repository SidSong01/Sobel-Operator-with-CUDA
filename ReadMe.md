[//]: # (Image References)

[image1]: ./outputs/cpu.jpg
[image2]: ./outputs/gpu1.jpg
[image3]: ./outputs/gpu2.jpg
[image4]: ./outputs/opencv.jpg
[image5]: ./test01.jpg
[image6]: ./outputs/screenshotResults.png

# TO DO:

 Include a quantified comparision, e.g., showing a table with norm difference between serail and parallel implementation. Improve code organization.

# Overview:
---
This is a project comparing the speed difference between the CPU and GPU implementation on Sobel Operator. For CPU implementation, OpenCV package and CPU computing on pixels have been tried. For GPU implementation, CUDA is used and different arrangement for the threads and blocks have been tried.

# How to run:
---

* Compile:

	`$ make`

* Run:

	`$./benchmarking.sh`

* Output information

![alt text][image6]

# Results:
---
* The test image: 

![alt text][image5]

Size: 599 x 393

* CPU with OpenCV package:

![alt text][image4]

* CPU on pixels:

![alt text][image1]

* GPU implementation with 1 thread/block and 1 block/grid:

![alt text][image2]

* GPU implementation with organized dimensions:

`blocks((int)((imgW+31)/32),(int)(g_imgHeight+31)/32);`
`threads(16, 16);`

![alt text][image3]

----

* Speed difference

| Method       		|     Execution Time (ms)	       | 
|:---------------------:|:---------------------------------------------:| 
| OpenCV package        | 1.700   							| 
| CPU on pixels	| 12.545	|
| GPU with single thread and block | 66.028 		|
| GPU with multiple threads and blocks | 0.528		|

