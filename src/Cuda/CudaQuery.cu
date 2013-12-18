#include "CudaQuery.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <thrust/iterator/constant_iterator.h>

#define CUDA_THREADS_PER_BLOCK 256

struct gpuElem
{
	int tag;
	int metric;
	unsigned long long int time;
	float value;
};

__global__ void cuda_produce_stencil(ddj::store::storeElement* elements, int elemCount, int* tags, int tagsCount, int* stencil)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= elemCount) return;
	int tag = elements[idx].tag;
	stencil[idx] = 0;
	while(tagsCount--)
	{
		if(tag == tags[tagsCount])
		{
			stencil[idx] = 1;
			return;
		}
	}

}

struct is_one
{
	__host__ __device__
	bool operator()(const int &x)
	{
		return x == 1;
	}
};

size_t gpu_filterData(ddj::store::storeElement* elements, int elemCount, ddj::store::storeQuery* query)
{
	// CREATE STENCIL
	int* stencil;
	cudaMalloc(&stencil, elemCount*sizeof(int));

	// CREATE TAGS VECTOR ON GPU
	thrust::device_vector<int> tags(query->tags.begin(), query->tags.end());

	// FILL STENCIL
	int blocksPerGrid =(elemCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	cuda_produce_stencil<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(elements, elemCount, tags.data().get(), tags.size(), stencil);

	// PARTITION ELEMENTS
	gpuElem* e = (gpuElem*)elements;
	thrust::partition(e, e+elemCount, stencil, is_one());

	// RETURN NUMBER OF ELEMENTS WITH TAG FROM QUERY'S TAGS
	return thrust::count(stencil, stencil+elemCount, 1);
}
