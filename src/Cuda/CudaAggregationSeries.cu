#include "CudaAggregation.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/partition.h>
#include <cmath>


// HOW TO PRINT STH TO CONSOLE IN KERNEL

// System includes
#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include "cuPrintf.cuh"

#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)

// CUPRINTF("\tIdx: %d, tag: %d, metric: %d, val: %f, Value is:%d\n", idx, tag, elements[idx].metric, elements[idx].value, 1);

// TODO: Move to config
#define MAX_METRIC_COUNT 65536

__global__ void cuda_produce_stencil_for_series(
		storeElement* elements,
		int elemCount,
		int tag,
		metric_type metric,
		int* stencil)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= elemCount) return;
	tag == elements[idx].tag && metric == elements[idx].metric ?
		stencil[idx] = 1 : stencil[idx] = 0;
}

__device__ int cuda_find_elements_for_interpolation(
		storeElement* elements,
		int elemCount,
		ullint time)
{
	int left = 0;
	int right = elemCount - 1;
	if(time < elements[left].time || time > elements[right].time) return -1;
	int middle;

	while(elemCount != 2)
	{
		middle = left + elemCount/2;

		if(time == elements[middle].time) return middle;

		if(time > elements[middle].time) left = middle;
		else right = middle;

		elemCount = right - left + 1;
	}
	return left;
}

__global__ void cuda_sum_series_with_interpolation(
		storeElement* elements,
		int elemCount,
		ullint* timePoints,
		int timePointCount,
		float* result)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= timePointCount) return;
	ullint time = timePoints[idx];
	int leftIndex = cuda_find_elements_for_interpolation(elements, elemCount, time);
	if(leftIndex == -1) return;
	float leftValue = elements[leftIndex].value;
	ullint leftTime = elements[leftIndex].time;
	float rightValue = elements[leftIndex+1].value;
	ullint rightTime = elements[leftIndex+1].time;
	if(leftTime == time) result[idx] += leftValue;
	else	// interpolate f(x) = y0 + (y1-y0)*(x-x0)/(x1-x0)
		result[idx] += leftValue + (rightValue-leftValue)*(time-leftTime)/(rightTime-leftTime);
}

size_t gpu_sum_series(storeElement* elements, size_t dataSize, void** result, ullint* timePoints,
		int timePointCount, metric_type* metrics, int metricCount, int* tags, int tagCount)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;
	int blocksPerGrid_stencil = (elemCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	int blocksPerGrid_interpolation = (timePointCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	int seriesElemCount = 0;
	thrust::device_ptr<storeElement> elem_ptr(elements);

	// ALLOCATE STENCIL
	int* stencil;
	cudaMalloc(&stencil, elemCount*sizeof(int));
	thrust::device_ptr<int> stencil_ptr(stencil);

	// ALLOCATE INTERPOLATION
	float* interpolation;
	cudaMalloc(&interpolation, sizeof(float)*timePointCount);
	cudaMemset(interpolation, 0, sizeof(float)*timePointCount);

	// ALLOCATE TIME POINTS ON DEVICE AND COPY THEM FROM HOST
	ullint* timePoints_device;
	cudaMalloc(&timePoints_device, sizeof(ullint)*timePointCount);
	cudaMemcpy(timePoints_device, timePoints, sizeof(ullint)*timePointCount, cudaMemcpyHostToDevice);

	for(int i=0; i<metricCount; i++)
	{
		for(int j=0; j<tagCount; j++)
		{
			// PRODUCE STENCIL FOR SERIES
			cuda_produce_stencil_for_series<<<blocksPerGrid_stencil, CUDA_THREADS_PER_BLOCK>>>(
						elements,
						elemCount,
						tags[j],
						metrics[i],
						stencil);
			cudaDeviceSynchronize();

			// MOVE SERIES TO FRONT
			thrust::stable_partition(
					elem_ptr,
					elem_ptr+elemCount,
					stencil_ptr,
					thrust::identity<int>());
			seriesElemCount = thrust::count(stencil_ptr, stencil_ptr+elemCount, 1);

			// INTERPOLATE AND ADD SERIES
			cuda_sum_series_with_interpolation<<<blocksPerGrid_interpolation, CUDA_THREADS_PER_BLOCK>>>(
					elements,
					seriesElemCount,
					timePoints_device,
					timePointCount,
					interpolation);
			cudaDeviceSynchronize();
		}
	}

	// COPY RESULT TO HOST
	float* interpolation_sum_host = new float[timePointCount];
	cudaMemcpy(interpolation_sum_host, interpolation, sizeof(float)*timePointCount, cudaMemcpyDeviceToHost);

	// CLEAN AND RETURN RESULT
	(*result) = interpolation_sum_host;
	cudaFree(stencil);
	cudaFree(interpolation);
	return sizeof(float)*timePointCount;
}
