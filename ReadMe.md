[//]: # (Image References)

[image1]: ./outputs/cpu.jpg
[image2]: ./outputs/gpu1.jpg
[image3]: ./outputs/gpu2.jpg
[image4]: ./outputs/opencv.jpg
[image5]: ./test01.jpg

# TO DO:

 Include a quantified comparision, e.g., showing a table with norm difference between serail and parallel implementation.

# Overview:

---

This is a project comparing the speed difference between the CPU and GPU implementation on Sobel Operator. For CPU implementation, OpenCV package and CPU computing on pixels have been tried. For GPU implementation, CUDA is used and different arrangement for the threads and blocks have been tried.

# Results:

---

* The test image is:

![alt text][image5]

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

# How to run:

---

* Compile:

	`$ make`

* Run:

	`$./benchmarking.sh`
