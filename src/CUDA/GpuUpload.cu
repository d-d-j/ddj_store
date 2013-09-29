#include "GpuUpload.cuh"

void callPrintData(void *data, int size) {
	printData<<<1, size>>>(data);
}

__host__ __device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

__global__ void printData(void *data) {
	int index = threadIdx.x;
	printf("data: %d, %f", index, (storeElement)data[index].value);

}
