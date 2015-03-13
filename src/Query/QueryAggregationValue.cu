#include "QueryAggregation.cuh"
#include "../Cuda/CudaIncludes.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cmath>


// HOW TO PRINT STH TO CONSOLE IN KERNEL

// System includes
#include <stdio.h>
#include <assert.h>
// CUDA runtime
#include <cuda_runtime.h>
#include "../Cuda/CudaPrintf.cuh"

#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)

// CUPRINTF("\tIdx: %d, tag: %d, metric: %d, val: %f, Value is:%d\n", idx, tag, elements[idx].metric, elements[idx].value, 1);

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

// STD DEVIATION OR VARIANCE, SKEWNESS, KURTOSIS

template <typename T>
struct variance_unary_op
{
	__host__ __device__
	results::varianceResult operator()(const T& x) const
	{
		results::varianceResult result;
		result.count = 1;
		result.mean = x.value;
		result.m2 = 0;
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
        result.m2 = x.m2 + y.m2;
        result.m2 += delta2 * x.count * y.count / count;

        return result;
    }
};

size_t gpu_variance(storeElement* elements, size_t dataSize, void** result)
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

template <typename T>
struct skewness_unary_op
{
	__host__ __device__
	results::skewnessResult operator()(const T& x) const
	{
		results::skewnessResult result;
		result.count = 1;
		result.mean = x.value;
		result.m2 = 0;
		result.m3 = 0;
		return result;
	}
};

struct skewness_binary_op
    : public thrust::binary_function<const results::skewnessResult&,
                                     const results::skewnessResult&,
                                     results::skewnessResult >
{
    __host__ __device__
    results::skewnessResult operator()(const results::skewnessResult& x, const results::skewnessResult& y) const
    {
    	results::skewnessResult result;

        float count  = x.count + y.count;
        float count2 = count  * count;

    	float delta  = y.mean - x.mean;
		float delta2 = delta  * delta;
		float delta3 = delta2 * delta;

    	result.count = count;
    	result.mean = x.mean + delta * y.count / count;

		result.m2  = x.m2 + y.m2;
		result.m2 += delta2 * x.count * y.count / count;

		result.m3  = x.m3 + y.m3;
		result.m3 += delta3 * x.count * y.count * (x.count - y.count) / count2;
		result.m3 += 3.0f * delta * (x.count * y.m2 - y.count * x.m2) / count;

        return result;
    }
};

size_t gpu_skewness(storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);

	skewness_unary_op<storeElement> unary_op;
	skewness_binary_op binary_op;
	results::skewnessResult init;

	results::skewnessResult* variance =
			new results::skewnessResult(thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, binary_op));
	(*result) = variance;

	return sizeof(results::skewnessResult);
}

template <typename T>
struct kurtosis_unary_op
{
	__host__ __device__
	results::kurtosisResult operator()(const T& x) const
	{
		results::kurtosisResult result;
		result.count = 1;
		result.mean = x.value;
		result.m2 = 0;
		result.m3 = 0;
		result.m4 = 0;
		return result;
	}
};

struct kurtosis_binary_op
    : public thrust::binary_function<const results::kurtosisResult&,
                                     const results::kurtosisResult&,
                                     results::kurtosisResult >
{
    __host__ __device__
    results::kurtosisResult operator()(const results::kurtosisResult& x, const results::kurtosisResult& y) const
    {
    	results::kurtosisResult result;

    	float count  = x.count + y.count;
    	float count2 = count  * count;
    	float count3 = count2 * count;

    	float delta  = y.mean - x.mean;
    	float delta2 = delta  * delta;
    	float delta3 = delta2 * delta;
    	float delta4 = delta3 * delta;

		result.count = count;

		result.mean = x.mean + delta * y.count / count;

		result.m2  = x.m2 + y.m2;
		result.m2 += delta2 * x.count * y.count / count;

		result.m3  = x.m3 + y.m3;
		result.m3 += delta3 * x.count * y.count * (x.count - y.count) / count2;
		result.m3 += 3.0f * delta * (x.count * y.m2 - y.count * x.m2) / count;

		result.m4  = x.m4 + y.m4;
		result.m4 += delta4 * x.count * y.count * (x.count * x.count - x.count * y.count + y.count * y.count) / count3;
		result.m4 += 6.0f * delta2 * (x.count * x.count * y.m2 + y.count * y.count * x.m2) / count2;
		result.m4 += 4.0f * delta * (x.count * y.m3 - y.count * x.m3) / count;

        return result;
    }
};

size_t gpu_kurtosis(storeElement* elements, size_t dataSize, void** result)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	thrust::device_ptr<storeElement> elem_ptr(elements);

	kurtosis_unary_op<storeElement> unary_op;
	kurtosis_binary_op binary_op;
	results::kurtosisResult init;

	results::kurtosisResult* variance =
			new results::kurtosisResult(thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, binary_op));
	(*result) = variance;

	return sizeof(results::kurtosisResult);
}

// TRUNK INTEGRAL

__global__ void calculate_trapezoid_fields(ddj::store::storeElement* elements, int count, float* result)
{
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;

	ullint timespan = elements[idx+1].time - elements[idx].time;
	result[idx] = ( elements[idx].value + elements[idx+1].value ) * timespan / 2;
	//CUPRINTF("\tIdx: %d, result: %f\n", idx, result[idx]);
}

__global__ void sum_fields_in_trunks(float* trapezoidFields, size_t elemSize, ddj::ullintPair* locations, int count, float* result)
{
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;

	int i = locations[idx].first/elemSize;
	int end = locations[idx].second/elemSize;
	float sum = 0;

	for(; i<end; i++)
	{
		sum += trapezoidFields[i];
	}
	result[idx] = sum;
}

__global__ void fill_integralResults(
		results::integralResult* result,
		storeElement* elements,
		size_t elemSize,
		float* integralSums,
		ddj::ullintPair* locations,
		int count)
{
	unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
	if(i >= count) return;
	result[i].integral = integralSums[i];
	int left = locations[i].first/elemSize;
	int right = locations[i].second/elemSize;
	result[i].left_value = elements[left].value;
	result[i].left_time= elements[left].time;
	result[i].right_value = elements[right].value;
	result[i].right_time= elements[right].time;
	//CUPRINTF("\tIdx: %d, integral: %f, left: %d/%llu, right: %d/%llu\n", i, integralSums[i], left, elements[left].time, right, elements[right].time);
}

size_t gpu_trunk_integral(storeElement* elements, size_t dataSize, void** result,
		ddj::ullintPair* dataLocationInfo, int locationInfoCount)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	// ALLOCATE SPACE FOR RESULTS
	float* trapezoidFields;
	float* integralSums;
	CUDA_CHECK_RETURN( cudaMalloc(&integralSums, sizeof(float)*locationInfoCount) );
	CUDA_CHECK_RETURN( cudaMalloc(&trapezoidFields, sizeof(float)*(elemCount-1)) );

	// CREATE TIME PERIODS VECTOR ON GPU
	thrust::device_vector<ddj::ullintPair> locations(dataLocationInfo, dataLocationInfo+locationInfoCount);

	// CALCULATE TRAPEZOID FIELDS
	int blocksPerGrid = (elemCount - 1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	calculate_trapezoid_fields<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			elements,
			elemCount-1,
			trapezoidFields);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

	// SUM UP FIELDS IN TRUNKS
	blocksPerGrid = (locationInfoCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	sum_fields_in_trunks<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				trapezoidFields,
				sizeof(storeElement),
				locations.data().get(),
				locationInfoCount,
				integralSums);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

	// CREATE RESULT
	results::integralResult* integral = new results::integralResult[locationInfoCount];
	results::integralResult* integral_on_device;
	CUDA_CHECK_RETURN( cudaMalloc((void**)&integral_on_device, sizeof(results::integralResult)*locationInfoCount) );
	fill_integralResults<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			integral_on_device,
			elements,
			storeElemSize,
			integralSums,
			locations.data().get(),
			locationInfoCount);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	CUDA_CHECK_RETURN( cudaMemcpy(integral, integral_on_device, sizeof(results::integralResult)*locationInfoCount, cudaMemcpyDeviceToHost) );

	// RETURN RESULT
	(*result)=integral;

	CUDA_CHECK_RETURN( cudaFree(integral_on_device) );
	CUDA_CHECK_RETURN( cudaFree(integralSums) );
	CUDA_CHECK_RETURN( cudaFree(trapezoidFields) );
	return locationInfoCount*sizeof(results::integralResult);
}

// HISTOGRAM

__device__ int find_bucket_value(float2* buckets, int bucketCount, float value)
{
	int leftIndex = 0;
	int rightIndex = bucketCount-1;
	int middleIndex;

	if(value < buckets[leftIndex].x || value >= buckets[rightIndex].y) return -1;
	if(bucketCount == 1) return 0;

	while(bucketCount != 2)
	{
		middleIndex = leftIndex+bucketCount/2;

		if(value >= buckets[middleIndex].y)
			leftIndex = middleIndex;
		else if(value < buckets[middleIndex].x)
			rightIndex = middleIndex;
		else
			return middleIndex;

		bucketCount = rightIndex - leftIndex + 1;
	}
	// bucketCount == 2
	if(value < buckets[leftIndex].y) return leftIndex;
	if(value < buckets[rightIndex].y) return rightIndex;
	return -1;
}

__device__ int find_bucket_time(ullint2* buckets, int bucketCount, ullint value)
{
	int leftIndex = 0;
	int rightIndex = bucketCount-1;
	int middleIndex;

	if(value < buckets[leftIndex].x || value >= buckets[rightIndex].y) return -1;
	if(bucketCount == 1) return 0;

	while(bucketCount != 2)
	{
		middleIndex = leftIndex+bucketCount/2;

		if(value >= buckets[middleIndex].y)
			leftIndex = middleIndex;
		else if(value < buckets[middleIndex].x)
			rightIndex = middleIndex;
		else
			return middleIndex;

		bucketCount = rightIndex - leftIndex + 1;
	}
	// bucketCount == 2
	if(value < buckets[leftIndex].y) return leftIndex;
	if(value < buckets[rightIndex].y) return rightIndex;
	return -1;
}

__global__ void calculate_histogram_value(storeElement* elements, int count, int* results, float2* buckets, int bucketCount)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;

	float value = elements[idx].value;
	int bucketNumber = find_bucket_value(buckets, bucketCount, value);
	if(bucketNumber != -1)
	{
		atomicAdd(results+bucketNumber,1);

	}
}

__global__ void calculate_histogram_time(storeElement* elements, int count, int* results, ullint2* buckets, int bucketCount)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;

	ullint value = elements[idx].time;
	int bucketNumber = find_bucket_time(buckets, bucketCount, value);
	if(bucketNumber != -1)
	{
		atomicAdd(results+bucketNumber,1);
	}
}

size_t gpu_histogram_value(storeElement* elements, size_t dataSize, void** result, float2* buckets, int bucketCount)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;
	int blocksPerGrid = (elemCount - 1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

	// ALLOCATE GPU MEMORY
	float2* buckets_device;
	int* histogram_device;
	CUDA_CHECK_RETURN( cudaMalloc(&buckets_device, sizeof(float2)*bucketCount) );
	CUDA_CHECK_RETURN( cudaMalloc(&histogram_device, sizeof(int)*bucketCount) );
	CUDA_CHECK_RETURN( cudaMemset(histogram_device, 0, sizeof(int)*bucketCount) );

	// COPY BUCKETS TO DEVICE
	CUDA_CHECK_RETURN( cudaMemcpy(buckets_device, buckets, sizeof(float2)*bucketCount, cudaMemcpyHostToDevice) );

	// LAUNCH KERNEL
	calculate_histogram_value<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			elements,
			elemCount,
			histogram_device,
			buckets_device,
			bucketCount
			);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

	// COPY HISTOGRAM TO CPU MEMORY
	int* histogram_host = new int[bucketCount];
	CUDA_CHECK_RETURN( cudaMemcpy(histogram_host, histogram_device, sizeof(int)*bucketCount, cudaMemcpyDeviceToHost) );

	// CLEAN UP
	CUDA_CHECK_RETURN( cudaFree( histogram_device ) );
	CUDA_CHECK_RETURN( cudaFree( buckets_device ) );

	//RETURN RESULT
	(*result) = histogram_host;
	return sizeof(int)*bucketCount;
}

size_t gpu_histogram_time(storeElement* elements, size_t dataSize, void** result, ullint2* buckets, int bucketCount)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;
	int blocksPerGrid = (elemCount - 1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

	// ALLOCATE GPU MEMORY
	ullint2* buckets_device;
	int* histogram_device;
	CUDA_CHECK_RETURN( cudaMalloc(&buckets_device, sizeof(ullint2)*bucketCount) );
	CUDA_CHECK_RETURN( cudaMalloc(&histogram_device, sizeof(int)*bucketCount) );
	CUDA_CHECK_RETURN( cudaMemset(histogram_device, 0, sizeof(int)*bucketCount) );

	// COPY BUCKETS TO DEVICE
	CUDA_CHECK_RETURN( cudaMemcpy(buckets_device, buckets, sizeof(ullint2)*bucketCount, cudaMemcpyHostToDevice) );

	// LAUNCH KERNEL
	calculate_histogram_time<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			elements,
			elemCount,
			histogram_device,
			buckets_device,
			bucketCount
			);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

	// COPY HISTOGRAM TO CPU MEMORY
	int* histogram_host = new int[bucketCount];
	CUDA_CHECK_RETURN( cudaMemcpy(histogram_host, histogram_device, sizeof(int)*bucketCount, cudaMemcpyDeviceToHost) );

	// CLEAN UP
	CUDA_CHECK_RETURN( cudaFree(histogram_device) );
	CUDA_CHECK_RETURN( cudaFree(buckets_device) );

	//RETURN RESULT
	(*result) = histogram_host;
	return sizeof(int)*bucketCount;
}
