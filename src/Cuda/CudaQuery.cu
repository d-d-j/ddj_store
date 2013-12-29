#include "CudaQuery.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>


// HOW TO PRINT STH TO CONSOLE IN KERNEL
//// System includes
//#include <stdio.h>
//#include <assert.h>
//// CUDA runtime
//#include <cuda_runtime.h>
//#include "cuPrintf.cu"
//#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
//                                  blockIdx.y*gridDim.x+blockIdx.x,\
//                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//                                  __VA_ARGS__)
//// CUPRINTF("\tIdx: %d, tag: %d, metric: %d, val: %f, Value is:%d\n", idx, tag, elements[idx].metric, elements[idx].value, 1);



#define CUDA_THREADS_PER_BLOCK 256

typedef struct
{
	int32_t tag;
	int metric;
	ullint time;
	float value;
} gpuElem;

__device__ bool isInside(ullint value, ddj::ullintPair* timePeriod)
{
	if(value >= timePeriod->first && value <= timePeriod->second) return true;
	else return false;
}

__global__ void cuda_produce_stencil_using_tag(
		ddj::store::storeElement* elements,
		int elemCount,
		int* tags,
		int tagsCount,
		int* stencil)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= elemCount) return;
	int32_t tag = elements[idx].tag;
	stencil[idx] = 0;
	while(tagsCount--)
	{
		if(tag == tags[tagsCount])
		{
			stencil[idx] = 1;
			return;
		}
	}
	return;
}

__global__ void cuda_produce_stencil_using_tagAndTime(
		ddj::store::storeElement* elements,
		int elemCount,
		int* tags,
		int tagsCount,
		ddj::ullintPair* timePeriods,
		int timePeriodsCount,
		int* stencil)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= elemCount) return;
	int32_t tag = elements[idx].tag;
	ullint time = elements[idx].time;
	stencil[idx] = 0;
	while(tagsCount--)
	{
		if(tag == tags[tagsCount])
		{
			while(timePeriodsCount--)
			{
				if(isInside(time, &timePeriods[timePeriodsCount]))
				{
					stencil[idx] = 1;
					return;
				}
			}
		}
	}
	return;
}

struct is_one
{
	__host__ __device__
	bool operator()(const int &x)
	{
		return x == 1;
	}
};

size_t gpu_filterData(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeQuery* query)
{
	// CREATE STENCIL
	int elemCount = dataSize/sizeof(ddj::store::storeElement);
	int* stencil;
	cudaMalloc(&stencil, elemCount*sizeof(int));

	// CREATE TAGS VECTOR ON GPU
	thrust::device_vector<int> tags(query->tags.begin(), query->tags.end());

	// FILL STENCIL
	int blocksPerGrid =(elemCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	if(query->timePeriods.size())
	{
		// CREATE TIME PERIODS VECTOR ON GPU
		thrust::device_vector<ddj::ullintPair> timePeriods(query->timePeriods.begin(), query->timePeriods.end());
		cuda_produce_stencil_using_tagAndTime<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				elements,
				elemCount,
				tags.data().get(),
				tags.size(),
				timePeriods.data().get(),
				timePeriods.size(),
				stencil);
	} else {
		cuda_produce_stencil_using_tag<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				elements,
				elemCount,
				tags.data().get(),
				tags.size(),
				stencil);
	}
	cudaDeviceSynchronize();

	// PARTITION ELEMENTS
	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	thrust::device_ptr<int> stencil_ptr(stencil);

	thrust::partition(thrust::device, elem_ptr, elem_ptr+elemCount, stencil, is_one());

	// RETURN NUMBER OF ELEMENTS WITH TAG FROM QUERY'S TAGS
	return thrust::count(stencil_ptr, stencil_ptr+elemCount, 1) * sizeof(ddj::store::storeElement);
}
