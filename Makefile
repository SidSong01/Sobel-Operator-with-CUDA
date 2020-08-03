objects=cpu gpu1 gpu2
edit:$(objects)

cpu:sobelWithCpu.cu
	@nvcc -g -o $@ $? `pkg-config opencv --cflags --libs`

gpu1:sobelWithNoOrganization.cu
	@nvcc -g -o $@ $? `pkg-config opencv --cflags --libs`

gpu2:sobelWithMul.cu
	@nvcc -g -o $@ $? `pkg-config opencv --cflags --libs`

