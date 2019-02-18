all: nms_kernel.cu nms_kernel.h
	nvcc -c -o nms.cu.o nms_kernel.cu --compiler-options -fPIC ${CUDA_ARCH}
clean:
	rm nms.cu.o
