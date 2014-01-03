#include "CudaQuery.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
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

size_t gpu_add_values(ddj::store::storeElement* elements, size_t dataSize, void** result)
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

size_t gpu_max_from_values(ddj::store::storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(ddj::store::storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	gpuElem init;
	cudaMemcpy(&init, elements, storeElemSize, cudaMemcpyDeviceToHost);
	gpuElem max = thrust::reduce(elem_ptr, elem_ptr+elemCount, init, max_gpu_elem());
	cudaMalloc(result, storeElemSize);
	cudaMemcpy(*result, &max, storeElemSize, cudaMemcpyDeviceToDevice);

	return storeElemSize;
}

size_t gpu_min_from_values(ddj::store::storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(ddj::store::storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	gpuElem init;
	cudaMemcpy(&init, elements, storeElemSize, cudaMemcpyDeviceToHost);
	gpuElem min = thrust::reduce(elem_ptr, elem_ptr+elemCount, init, min_gpu_elem());
	cudaMalloc(result, storeElemSize);
	cudaMemcpy(*result, &min, storeElemSize, cudaMemcpyDeviceToDevice);

	return storeElemSize;
}

size_t gpu_average_from_values(ddj::store::storeElement* elements, size_t dataSize, void** result)
{
	size_t averageSize = sizeof(ddj::query::results::averageResult);
	int elemCount = dataSize / sizeof(ddj::store::storeElement);

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);
	gpuElem init;
	init.value = 0;
	gpuElem sum = thrust::reduce(elem_ptr, elem_ptr+elemCount, init, sum_gpu_elem());
	ddj::query::results::averageResult* average = new ddj::query::results::averageResult(sum.value, elemCount);
	(*result) = average;
	return averageSize;
}

template <typename T>
struct variance_gpu_elem
{
	store_value_type n;
	store_value_type mean;
	store_value_type M2;

    void initialize() { n = mean = M2 = 0; }
    T variance() { return T(0,0,0, M2 / (n - 1) ); }
    T variance_n() { return T(0,0,0, M2 / n ); }
    T stdDeviation()
    {
    	return T(0,0,0, std::sqrt(M2 / (n - 1)) );
    }
};

template <typename T>
struct variance_unary_op
{
	__host__ __device__
	variance_gpu_elem<T> operator()(const T& x) const
	{
		variance_gpu_elem<T> result;
		result.n = 1;
		result.mean = x.value;
		result.M2 = 0;
		return result;
	}
};

template <typename T>
struct variance_binary_op
    : public thrust::binary_function<const variance_gpu_elem<T>&,
                                     const variance_gpu_elem<T>&,
                                     variance_gpu_elem<T> >
{
    __host__ __device__
    variance_gpu_elem<T> operator()(const variance_gpu_elem<T>& x, const variance_gpu_elem <T>& y) const
    {
    	variance_gpu_elem<T> result;

    	store_value_type n = x.n + y.n;
    	store_value_type delta = y.mean - x.mean;
    	store_value_type delta2 = delta * delta;
        result.n = n;
        result.mean = x.mean + delta * y.n / n;
        result.M2 = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};

size_t gpu_stdDeviation_from_values(ddj::store::storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(ddj::store::storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<gpuElem> elem_ptr((gpuElem*)elements);

	variance_unary_op<gpuElem> unary_op;
	variance_binary_op<gpuElem> binary_op;
	variance_gpu_elem<gpuElem> init;
	init.initialize();

	variance_gpu_elem<gpuElem> variance = thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, binary_op);
	gpuElem stdDev = variance.stdDeviation();

	cudaMalloc(result, storeElemSize);
	cudaMemcpy(*result, &stdDev, storeElemSize, cudaMemcpyHostToDevice);

	return storeElemSize;
}
