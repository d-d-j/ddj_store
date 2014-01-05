#include "CudaQuery.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>


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


using namespace ddj::store;

// TODO: Remove repeating code

struct is_one
{
	__host__ __device__
	bool operator()(const int &x)
	{
		return x == 1;
	}
};

__device__ bool isInside(ullint value, ddj::ullintPair* timePeriod)
{
	if(value >= timePeriod->first && value <= timePeriod->second) return true;
	else return false;
}

__global__ void cuda_produce_stencil_using_tag(
		storeElement* elements,
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

__global__ void cuda_produce_stencil_using_time(
		storeElement* elements,
		int elemCount,
		ddj::ullintPair* timePeriods,
		int timePeriodsCount,
		int* stencil)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= elemCount) return;
	ullint time = elements[idx].time;
	stencil[idx] = 0;
	while(timePeriodsCount--)
	{
		if(isInside(time, &timePeriods[timePeriodsCount]))
		{
			stencil[idx] = 1;
			return;
		}
	}
	return;
}

__global__ void cuda_produce_stencil_using_tag_and_time(
		storeElement* elements,
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

__global__ void sum_stencil_in_trunks(int* stencil, size_t elemSize, ddj::ullintPair* locations, int count, int* result)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;
	int i = locations[idx].first/elemSize;
	int end = locations[idx].second/elemSize;
	int sum = 0;
	for(; i<=end; i++)
	{
		sum += stencil[i];
	}
	result[idx] = sum;
}

int* gpu_produceStencil(storeElement* elements, size_t dataSize, ddj::query::Query* query)
{
	int elemCount = dataSize/sizeof(storeElement);
	int* stencil;
	cudaMalloc(&stencil, elemCount*sizeof(int));
	int blocksPerGrid =(elemCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

	// CREATE TIME PERIODS VECTOR ON GPU
	thrust::device_vector<ddj::ullintPair> timePeriods(query->timePeriods.begin(), query->timePeriods.end());
	// CREATE TAGS VECTOR ON GPU
	thrust::device_vector<int> tags(query->tags.begin(), query->tags.end());

	// RUN STENCIL KERNEL
	int filterTags = query->tags.size();
	int filterTimePeriods = query->timePeriods.size();
	if(filterTags && filterTimePeriods)
	{
		cuda_produce_stencil_using_tag_and_time<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				elements,
				elemCount,
				tags.data().get(),
				tags.size(),
				timePeriods.data().get(),
				timePeriods.size(),
				stencil);
	} else if(filterTags){
		cuda_produce_stencil_using_tag<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				elements,
				elemCount,
				tags.data().get(),
				tags.size(),
				stencil);
	} else {
		cuda_produce_stencil_using_time<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				elements,
				elemCount,
				timePeriods.data().get(),
				timePeriods.size(),
				stencil);
	}
	cudaDeviceSynchronize();
	return stencil;
}

size_t gpu_filterData(storeElement* elements, size_t dataSize, ddj::query::Query* query)
{
	int elemCount = dataSize/sizeof(storeElement);

	// CREATE STENCIL
	int* stencil = gpu_produceStencil(elements, dataSize, query);

	// PARTITION ELEMENTS
	thrust::device_ptr<storeElement> elem_ptr(elements);
	thrust::device_ptr<int> stencil_ptr(stencil);

	thrust::partition(thrust::device, elem_ptr, elem_ptr+elemCount, stencil, is_one());

	// RETURN NUMBER OF ELEMENTS WITH TAG FROM QUERY'S TAGS
	size_t resultSize = thrust::count(stencil_ptr, stencil_ptr+elemCount, 1) * sizeof(storeElement);
	cudaFree(stencil);
	return resultSize;
}

size_t gpu_filterData_in_trunks(storeElement* elements, size_t dataSize, Query* query,
				ddj::ullintPair* dataLocationInfo, int locationInfoCount)
{
	int elemCount = dataSize/sizeof(storeElement);

	// CREATE STENCIL
	int* stencil = gpu_produceStencil(elements, dataSize, query);

	// CREATE TIME PERIODS VECTOR ON GPU
	thrust::device_vector<ddj::ullintPair> locations(dataLocationInfo, dataLocationInfo+locationInfoCount);

	// PARTITION ELEMENTS
	thrust::device_ptr<storeElement> elem_ptr(elements);
	thrust::device_ptr<int> stencil_ptr(stencil);

	thrust::stable_partition(thrust::device, elem_ptr, elem_ptr+elemCount, stencil, is_one());

	// COUNT ELEMENTS IN TRUNKS
	int* trunkElemCount_device;
	cudaMalloc(&trunkElemCount_device, locationInfoCount*sizeof(int));
	int blocksPerGrid = (locationInfoCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	sum_stencil_in_trunks<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			stencil,
			sizeof(storeElement),
			locations.data().get(),
			locationInfoCount,
			trunkElemCount_device);
	cudaDeviceSynchronize();

	// DOWNLOAD TRUNK ELEM COUNT TO HOST
	int* trunkElemCount_host = new int[locationInfoCount];
	cudaMemcpy(trunkElemCount_host, trunkElemCount_device, sizeof(int)*locationInfoCount, cudaMemcpyDeviceToHost);
	cudaFree(trunkElemCount_device);

	// SET NEW DATA LOCATION INFO
	int position = 0;
	for(int i=0; i < locationInfoCount; i++)
	{
		dataLocationInfo[i].first = position;
		position += trunkElemCount_host[i]*sizeof(storeElement);
		dataLocationInfo[i].second = position - 1;
	}
	delete [] trunkElemCount_host;
	size_t resultSize = thrust::count(stencil_ptr, stencil_ptr+elemCount, 1) * sizeof(storeElement);
	cudaFree(stencil);
	return resultSize;
}
