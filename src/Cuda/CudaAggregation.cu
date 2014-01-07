#include "CudaAggregation.cuh"
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
#include "cuPrintf.cuh"

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

// STD DEVIATION AND VARIANCE

template <typename T>
struct variance_unary_op
{
	__host__ __device__
	results::statisticResult operator()(const T& x) const
	{
		results::statisticResult result;
		result.count = 1;
		result.mean = x.value;
		result.factor = 0;
		return result;
	}
};

struct variance_binary_op
    : public thrust::binary_function<const results::statisticResult&,
                                     const results::statisticResult&,
                                     results::statisticResult >
{
    __host__ __device__
    results::statisticResult operator()(const results::statisticResult& x, const results::statisticResult& y) const
    {
    	results::statisticResult result;

    	float count = x.count + y.count;
    	float delta = y.mean - x.mean;
    	float delta2 = delta * delta;
        result.count = count;
        result.mean = x.mean + delta * y.count / count;
        result.factor = x.factor + y.factor;
        result.factor += delta2 * x.count * y.count / count;

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
	results::statisticResult init;

	results::statisticResult* variance =
			new results::statisticResult(thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, binary_op));
	(*result) = variance;

	return sizeof(results::statisticResult);
}

// TRUNK INTEGRAL

__global__ void calculate_trapezoid_fields(ddj::store::storeElement* elements, int count, float* result)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;

	ullint timespan = elements[idx+1].time - elements[idx].time;
	result[idx] = ( elements[idx].value + elements[idx+1].value ) * timespan / 2;
}

__global__ void sum_fields_in_trunks(float* fields, size_t elemSize, ddj::ullintPair* locations, int count, float* result)
{
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(idx >= count) return;

	int i = locations[idx].first/elemSize;
	int end = locations[idx].second/elemSize;
	float sum = 0;
	for(; i<end; i++)
	{
		sum += fields[i];
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
	CUPRINTF("\tIdx: %d, integral: %f, left: %d/%llu, right: %d/%llu\n", i, integralSums[i], left, elements[left].time, right, elements[right].time);
}

size_t gpu_trunk_integral(storeElement* elements, size_t dataSize, void** result,
		ddj::ullintPair* dataLocationInfo, int locationInfoCount)
{
	size_t storeElemSize = sizeof(storeElement);
	int elemCount = dataSize / storeElemSize;

	// ALLOCATE SPACE FOR RESULTS
	float* integralSums;
	cudaMalloc(&integralSums, sizeof(float)*locationInfoCount);
	float* trapezoidFields;
	cudaMalloc(&trapezoidFields, sizeof(float)*(elemCount-1));

	// CREATE TIME PERIODS VECTOR ON GPU
	thrust::device_vector<ddj::ullintPair> locations(dataLocationInfo, dataLocationInfo+locationInfoCount);

	// CALCULATE TRAPEZOID FIELDS
	int blocksPerGrid = (elemCount - 1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	calculate_trapezoid_fields<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(elements, elemCount-1, trapezoidFields);
	if(cudaSuccess != cudaDeviceSynchronize()) printf("\n\nERROR\n\n");

	// SUM UP FIELDS IN TRUNKS
	blocksPerGrid = (locationInfoCount + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	sum_fields_in_trunks<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
				trapezoidFields,
				sizeof(storeElement),
				locations.data().get(),
				locationInfoCount,
				integralSums);
	if(cudaSuccess != cudaDeviceSynchronize()) printf("\n\nERROR\n\n");

	// CREATE RESULT
	results::integralResult* integral = new results::integralResult[locationInfoCount];
	results::integralResult* integral_on_device;
	cudaMalloc((void**)&integral_on_device, sizeof(results::integralResult)*locationInfoCount);
	fill_integralResults<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			integral_on_device,
			elements,
			storeElemSize,
			integralSums,
			locations.data().get(),
			locationInfoCount);
	if(cudaSuccess != cudaDeviceSynchronize()) printf("ERROR");
	if(cudaSuccess != cudaMemcpy(integral, integral_on_device, sizeof(results::integralResult)*locationInfoCount, cudaMemcpyDeviceToHost)) printf("ERROR");

	for(int i=0; i<locationInfoCount; i++)
	{
		printf("Result[%d] = int: %f, left: %ld, right: %ld\n", i, integral[i].integral, integral[i].left_time, integral[i].right_time);
	}

	// RETURN RESULT
	(*result)=integral;

	cudaFree(integral_on_device);
	cudaFree(integralSums);
	cudaFree(trapezoidFields);
	return locationInfoCount*sizeof(results::integralResult);
}

// HISTOGRAM

__device__ int find_bucket_value(float2* buckets, int bucketCount, float value)
{
	if(bucketCount == 1) return 0;

	int leftIndex = 0;
	int rightIndex = bucketCount-1;
	int middleIndex;

	if(value < buckets[leftIndex].x || value > buckets[rightIndex].y) return -1;


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
	if(bucketCount == 1) return 0;

	int leftIndex = 0;
	int rightIndex = bucketCount-1;
	int middleIndex;

	if(value < buckets[leftIndex].x || value > buckets[rightIndex].y) return -1;


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
	cudaMalloc(&buckets_device, sizeof(float2)*bucketCount);
	int* histogram_device;
	cudaMalloc(&histogram_device, sizeof(int)*bucketCount);
	cudaMemset(histogram_device, 0, sizeof(int)*bucketCount);

	// COPY BUCKETS TO DEVICE
	cudaMemcpy(buckets_device, buckets, sizeof(float2)*bucketCount, cudaMemcpyHostToDevice);

	// LAUNCH KERNEL
	calculate_histogram_value<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			elements,
			elemCount,
			histogram_device,
			buckets_device,
			bucketCount
			);
	cudaDeviceSynchronize();
	//if(cudaSuccess != cudaGetLastError()) throw std::runtime_error("calculate_histogram_value kernel error");

	// COPY HISTOGRAM TO CPU MEMORY
	int* histogram_host = new int[bucketCount];
	cudaMemcpy(histogram_host, histogram_device, sizeof(int)*bucketCount, cudaMemcpyDeviceToHost);

	// CLEAN UP
	cudaFree( histogram_device );
	cudaFree( buckets_device );

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
	cudaMalloc(&buckets_device, sizeof(ullint2)*bucketCount);
	int* histogram_device;
	cudaMalloc(&histogram_device, sizeof(int)*bucketCount);
	cudaMemset(histogram_device, 0, sizeof(int)*bucketCount);

	// COPY BUCKETS TO DEVICE
	cudaMemcpy(buckets_device, buckets, sizeof(ullint2)*bucketCount, cudaMemcpyHostToDevice);

	// LAUNCH KERNEL
	calculate_histogram_time<<<blocksPerGrid, CUDA_THREADS_PER_BLOCK>>>(
			elements,
			elemCount,
			histogram_device,
			buckets_device,
			bucketCount
			);
	cudaDeviceSynchronize();
	//if(cudaSuccess != cudaGetLastError()) throw std::runtime_error("calculate_histogram_time kernel error");

	// COPY HISTOGRAM TO CPU MEMORY
	int* histogram_host = new int[bucketCount];
	cudaMemcpy(histogram_host, histogram_device, sizeof(int)*bucketCount, cudaMemcpyDeviceToHost);

	// CLEAN UP
	cudaFree( histogram_device );
	cudaFree( buckets_device );

	//RETURN RESULT
	(*result) = histogram_host;
	return sizeof(int)*bucketCount;
}







