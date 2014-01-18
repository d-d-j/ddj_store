#include "CompressionLightweight.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

union converter {
	int64_t toInt64, fromInt64;
	int32_t toInt32, fromInt32;
	float toFloat, fromFloat;
	unsigned char toBytes[8], fromBytes[8];
};

__host__
__device__
void copyBytes(unsigned char * dest, const unsigned char * source, const int size) {
	for (int i = 0; i < size; i++) {
		dest[i] = source[i];
	}
}

__host__
__device__
void EncodeInt32UsingNBytes(unsigned char* out, int32_t value, int N)
{
	converter c;
	c.fromInt32 = value;
	copyBytes(out, c.toBytes, N);
}

__host__
__device__
void EncodeInt64UsingNBytes(unsigned char* out, int64_t value, int N)
{
	converter c;
	c.fromInt64 = value;
	copyBytes(out, c.toBytes, N);
}

__global__
void EncodeKernel(storeElement * in_d, unsigned char * out_d) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t low = in_d[index].time & 0xFFFFFFFF;
	int32_t high = (in_d[index].time >> 32) & 0xFFFFFFFF;
	int32_t position = 10 * index + 4;
	converter c;
	if (index == 0) {
		c.fromInt32 = high;
		copyBytes(out_d, c.toBytes, 4);
	}
	out_d[position] = (unsigned char) in_d[index].tag;
	position++;
	out_d[position] = (unsigned char) in_d[index].metric;
	position++;

	c.fromInt32 = low;
	copyBytes(out_d + position, c.toBytes, 4);

	position += 4;

	c.fromFloat = in_d[index].value;
	copyBytes(out_d + position, c.toBytes, 4);
}

__global__
void DecodeKernel(unsigned char * in_d, storeElement * out_d) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	converter c;
	copyBytes(c.fromBytes, in_d, 4);

	int64_t high = c.toInt32;
	int32_t position = 10 * index + 4;

	out_d[index].tag = in_d[position];
	position++;
	out_d[index].metric = in_d[position];
	position++;

	copyBytes(c.fromBytes, in_d + position, 4);
	out_d[index].time = ((int64_t) c.toInt32 & 0xFFFFFFFF) | (high << 32);

	position += 4;

	copyBytes(c.fromBytes, in_d + position, 4);
	out_d[index].value = c.toFloat;
}

size_t CompressLightweight(storeElement* elements, size_t size, unsigned char** result)
{
	int elemCount = size / sizeof(storeElement);

	//prepare compression output
	unsigned char *compressionOutput_device;	//output space for compressed data
	cudaMalloc((void**) &compressionOutput_device, COMPRESSED_DATA_SIZE(elemCount));
	cudaMemset(compressionOutput_device, 0, COMPRESSED_DATA_SIZE(elemCount));

	//compress
	int blocks = (elemCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	EncodeKernel<<<blocks, THREADS_PER_BLOCK>>>(elements, compressionOutput_device);
	cudaDeviceSynchronize();

	//return result
	(*result) = compressionOutput_device;
	return COMPRESSED_DATA_SIZE(elemCount);
}

size_t DecompressLightweight(unsigned char* data, size_t size, storeElement** result)
{
	int elemCount = size / COMPRESSED_ELEMENT_SIZE;

	// prepare decompression output
	storeElement* decompressionOutput_device;	//output space for decompressed data
	cudaMalloc((void**) &decompressionOutput_device, elemCount * sizeof(storeElement));
	cudaMemset(decompressionOutput_device, 0, elemCount * sizeof(storeElement));

	//decompress
	int blocks = (elemCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	unsigned char *decompessionInput_device = data;
	DecodeKernel<<<blocks, THREADS_PER_BLOCK>>>(decompessionInput_device, decompressionOutput_device);
	cudaDeviceSynchronize();

	(*result) = decompressionOutput_device;
	return elemCount * sizeof(storeElement);
}

struct trunkMinMax
{
	int32_t tag_min;
	int32_t tag_max;
	int32_t metric_min;
	int32_t metric_max;
	int64_t time_min;
	int64_t time_max;

	void initialize()
	{
		tag_min = std::numeric_limits<int32_t>::max();
		tag_max = std::numeric_limits<int32_t>::min();
		metric_min = std::numeric_limits<int32_t>::max();
		metric_max = std::numeric_limits<int32_t>::min();
		time_min = std::numeric_limits<int64_t>::max();
		time_max = std::numeric_limits<int64_t>::min();
	}
};

template <typename T>
struct trunkMinMax_unary_op
{
	__host__ __device__
	trunkMinMax operator()(const T& x) const
	{
		trunkMinMax result;

		result.tag_min = x.tag;
		result.tag_max = x.tag;

		result.metric_min = x.metric;
		result.metric_max = x.metric;

		result.time_min = x.time;
		result.time_max = x.time;

		return result;
	}
};

struct trunkMinMax_binary_op : public thrust::binary_function<const trunkMinMax&, const trunkMinMax&, trunkMinMax>
{
    __host__ __device__
    trunkMinMax operator()(const trunkMinMax& x, const trunkMinMax& y) const
    {
    	trunkMinMax result;

    	result.tag_min = thrust::min(x.tag_min, y.tag_min);
		result.tag_max = thrust::max(x.tag_max, y.tag_max);

		result.metric_min = thrust::min(x.metric_min, y.metric_min);
    	result.metric_max = thrust::max(x.metric_max, y.metric_max);

    	result.time_min = thrust::min(x.time_min, y.time_min);
		result.time_max = thrust::max(x.time_max, y.time_max);

		return result;
    }
};

unsigned int EasyFindLog2(int32_t v)
{
	unsigned int r = 0;

	while (v >>= 1)
	{
	  r++;
	}

	return r;
}

unsigned int FastFindLog2(int32_t v)
{
	const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	const unsigned int S[] = {1, 2, 4, 8, 16};
	int i;
	register unsigned int r = 0;

	for (i = 4; i >= 0; i--)
	{
	  if (v & b[i])
	  {
	    v >>= S[i];
	    r |= S[i];
	  }
	}

	return r;
}

template <typename T>
struct trunkCompressInfo_unary_op
{
	const trunkCompressInfo info;

	trunkCompressInfo_unary_op(trunkCompressInfo _info) : info(_info) {}

	__host__ __device__
	T operator()(const T& x) const
	{
		T newElement(x.tag-info.tag_min, x.metric-info.metric_min, x.time-info.time_min, x.value);
		return newElement;
	}
};

void PrepareElementsForCompression(storeElement* elements, int elemCount, trunkCompressInfo info)
{
	thrust::device_ptr<storeElement> elem_ptr(elements);
	trunkCompressInfo_unary_op<storeElement> unary_op(info);

	// Transform elements
	thrust::transform(elem_ptr, elem_ptr+elemCount, elem_ptr, unary_op);
}

trunkCompressInfo AnalizeTrunkData(storeElement* elements, int elemCount)
{
	// Produce min and max of trunk data
	thrust::device_ptr<storeElement> elem_ptr(elements);
	trunkMinMax_unary_op<storeElement> unary_op;
	trunkMinMax_binary_op binary_op;
	trunkMinMax init;
	init.initialize();
	trunkMinMax minMax = thrust::transform_reduce(elem_ptr, elem_ptr+elemCount, unary_op, init, binary_op);

	// Calculate bytes needed for each storeElement field
	int tag_bytes = (EasyFindLog2(minMax.tag_max - minMax.tag_min) + 7) / 8;
	int metric_bytes = (EasyFindLog2(minMax.metric_max - minMax.metric_min) + 7) / 8;
	int64_t timeDifference = minMax.time_max - minMax.time_min;
	if(timeDifference > std::numeric_limits<int32_t>::max())
		throw std::runtime_error("time difference in trunk is too big!");
	int time_bytes = (EasyFindLog2((int32_t)(timeDifference)) + 7) / 8;

	// Fill trunk compress info
	trunkCompressInfo info;
	info.tag_min = minMax.tag_min;
	info.metric_min = minMax.metric_min;
	info.time_min = minMax.time_min;
	info.bytes = ((tag_bytes & 0x000000FF) << 24) | ((metric_bytes & 0x000000FF) << 16) | (time_bytes & 0x0000FFFF);

	return info;
}
