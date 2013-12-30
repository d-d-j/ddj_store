#include "CudaQuery.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

struct sum_gpu_elem
{
    __host__ __device__
        gpuElem operator()(const gpuElem &lhs, const gpuElem &rhs) const
    {
    	gpuElem result;
    	result.value = lhs.value+rhs.value;
    	return result;
    }
};

struct min_gpu_elem
{
    __host__ __device__
        gpuElem operator()(const gpuElem &lhs, const gpuElem &rhs) const
    {
    	return lhs.value < rhs.value ? lhs : rhs;
    }
};

struct max_gpu_elem
{
    __host__ __device__
        gpuElem operator()(const gpuElem &lhs, const gpuElem &rhs) const
    {
    	return lhs.value < rhs.value ? rhs : lhs;
    }
};

size_t gpu_add_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result)
{
	size_t storeElemSize = sizeof(ddj::store::storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	gpuElem init;
	init.value = 0;
	gpuElem sum = thrust::reduce(elem_ptr, elem_ptr+elemCount, init, sum_gpu_elem());
	cudaMalloc(result, storeElemSize);
	cudaMemcpy(*result, &sum, storeElemSize, cudaMemcpyDeviceToDevice);

	return storeElemSize;
}

size_t gpu_max_from_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result)
{
	size_t storeElemSize = sizeof(ddj::store::storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	gpuElem init = elements[0];
	gpuElem max = thrust::reduce(elem_ptr, elem_ptr+elemCount, init, max_gpu_elem());
	cudaMalloc(result, storeElemSize);
	cudaMemcpy(*result, &max, storeElemSize, cudaMemcpyDeviceToDevice);

	return storeElemSize;
}

size_t gpu_min_from_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result)
{
	size_t storeElemSize = sizeof(ddj::store::storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	gpuElem init = elements[0];
	gpuElem min = thrust::reduce(elem_ptr, elem_ptr+elemCount, init, min_gpu_elem());
	cudaMalloc(result, storeElemSize);
	cudaMemcpy(*result, &min, storeElemSize, cudaMemcpyDeviceToDevice);

	return storeElemSize;
}
