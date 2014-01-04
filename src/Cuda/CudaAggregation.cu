#include "CudaQuery.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cmath>

// HOW TO PRINT STH TO CONSOLE IN KERNEL
/*
// System includes
#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include "cuPrintf.cu"
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)
// CUPRINTF("\tIdx: %d, tag: %d, metric: %d, val: %f, Value is:%d\n", idx, tag, elements[idx].metric, elements[idx].value, 1);
*/

// MIN AND MAX

struct min_gpu_elem
{
    __host__ __device__
        storeElement operator()(const storeElement &lhs, const storeElement &rhs) const
    {
    	return lhs.value < rhs.value ? lhs : rhs;
    }
};

struct max_gpu_elem
{
    __host__ __device__
        storeElement operator()(const storeElement &lhs, const storeElement &rhs) const
    {
    	return lhs.value < rhs.value ? rhs : lhs;
    }
};

size_t gpu_max(storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);

	storeElement init;
	cudaMemcpy(&init, elements, storeElemSize, cudaMemcpyDeviceToHost);

	storeElement* max =
			new storeElement(thrust::reduce(elem_ptr, elem_ptr+elemCount, init, max_gpu_elem()));
	(*result) = max;

	return storeElemSize;
}

size_t gpu_min(storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);
	storeElement init;
	cudaMemcpy(&init, elements, storeElemSize, cudaMemcpyDeviceToHost);

	storeElement* min =
			new storeElement(thrust::reduce(elem_ptr, elem_ptr+elemCount, init, min_gpu_elem()));
	(*result) = min;

	return storeElemSize;
}

// SUM AND AVERAGE

template <typename T>
struct sum_unary_op
{
	__host__ __device__
	float operator()(const T& x) const
	{
		return x.value;
	}
};

size_t gpu_sum(storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);

	float init = 0.0f;
	sum_unary_op<storeElement> unary_op;

	results::sumResult* sum =
			new results::sumResult(thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, thrust::plus<float>()));
	(*result) = sum;

	return sizeof(results::sumResult);
}

size_t gpu_average(ddj::store::storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);

	float init = 0.0f;
	sum_unary_op<storeElement> unary_op;

	results::averageResult* average =
			new results::averageResult(thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, thrust::plus<float>()), elemCount);
	(*result) = average;

	return sizeof(results::averageResult);
}

// STD DEVIATION AND VARIANCE

template <typename T>
struct variance_unary_op
{
	__host__ __device__
	results::varianceResult operator()(const T& x) const
	{
		results::varianceResult result;
		result.count = 1;
		result.mean = x.value;
		result.M2 = 0;
		return result;
	}
};

struct variance_binary_op
    : public thrust::binary_function<const results::varianceResult&,
                                     const results::varianceResult&,
                                     results::varianceResult >
{
    __host__ __device__
    results::varianceResult operator()(const results::varianceResult& x, const results::varianceResult& y) const
    {
    	results::varianceResult result;

    	float count = x.count + y.count;
    	float delta = y.mean - x.mean;
    	float delta2 = delta * delta;
        result.count = count;
        result.mean = x.mean + delta * y.count / count;
        result.M2 = x.M2 + y.M2;
        result.M2 += delta2 * x.count * y.count / count;

        return result;
    }
};

size_t gpu_stdDeviation(storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);

	variance_unary_op<storeElement> unary_op;
	variance_binary_op binary_op;
	results::varianceResult init;

	results::varianceResult* variance =
			new results::varianceResult(thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, binary_op));
	(*result) = variance;

	return sizeof(results::varianceResult);
}
