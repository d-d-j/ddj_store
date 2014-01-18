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

__host__ __device__
void copyBytes(unsigned char * dest, const unsigned char * source, const int size) {
	for (int i = 0; i < size; i++) {
		dest[i] = source[i];
	}
}

__host__ __device__
void EncodeInt32UsingNBytes(unsigned char* out, int32_t value, int N)
{
	converter c;
	c.fromInt32 = value;
	copyBytes(out, c.toBytes, N);
}

__host__ __device__
void EncodeInt64UsingNBytes(unsigned char* out, int64_t value, int N)
{
	converter c;
	c.fromInt64 = value;
	copyBytes(out, c.toBytes, N);
}

__global__
void EncodeKernel(
		storeElement * in_d,
		int elemCount,
		unsigned char * out_d,
		int tag_bytes,
		int metric_bytes,
		int time_bytes,
		int compressedElemSize,
		trunkCompressInfo info)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= elemCount) return;

	int32_t position = index * compressedElemSize;
	converter c;
	// tag
	int32_t tag = in_d[index].tag - info.tag_min;
	EncodeInt32UsingNBytes(out_d+position, tag, tag_bytes);
	position += tag_bytes;
	// metric
	int32_t metric = in_d[index].metric - info.metric_min;
	EncodeInt32UsingNBytes(out_d+position, metric, metric_bytes);
	position += metric_bytes;
	// time
	int64_t time = in_d[index].time - info.time_min;
	EncodeInt64UsingNBytes(out_d+position, time, time_bytes);
	position += time_bytes;
	// value
	c.fromFloat = in_d[index].value;
	copyBytes(out_d + position, c.toBytes, 4);
}

__global__
void DecodeKernel(
		unsigned char * in_d,
		int elemCount,
		storeElement * out_d,
		int tag_bytes,
		int metric_bytes,
		int time_bytes,
		int compressedElemSize,
		trunkCompressInfo info)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= elemCount) return;

	int32_t position = compressedElemSize * index;
	converter c;
	//tag
	c.toInt32 = 0;
	copyBytes(c.fromBytes, in_d+position, tag_bytes);
	out_d[index].tag = c.toInt32 + info.tag_min;
	position += tag_bytes;
	//metric
	c.toInt32 = 0;
	copyBytes(c.fromBytes, in_d+position, metric_bytes);
	out_d[index].metric = c.toInt32 + info.metric_min;
	position += metric_bytes;
	//time
	c.toInt64 = 0;
	copyBytes(c.fromBytes, in_d+position, time_bytes);
	out_d[index].time = c.toInt64 + info.time_min;
	position += time_bytes;
	//value
	copyBytes(c.fromBytes, in_d+position, 4);
	out_d[index].value = c.toFloat;
}

size_t CompressLightweight(storeElement* elements, size_t size, unsigned char** result)
{
	int elemCount = size / sizeof(storeElement);
	trunkCompressInfo info = AnalizeTrunkData(elements, elemCount);

	int tagBytes = (info.bytes & 0xFF000000) >> 24;
	int metricBytes = (info.bytes & 0x00FF0000) >> 16;
	int timeBytes = info.bytes & 0x0000FFFF;
	size_t compressedElemSize = tagBytes + metricBytes + timeBytes + 4;

	// prepare compression output
	unsigned char *compressionOutput_device;	//output space for compressed data
	size_t outputSize = compressedElemSize*elemCount+sizeof(trunkCompressInfo);
	cudaMalloc((void**) &compressionOutput_device, outputSize);
	cudaMemset(compressionOutput_device, 0, outputSize);
	cudaMemcpy(compressionOutput_device, &info, sizeof(trunkCompressInfo), cudaMemcpyHostToDevice);

	// compress
	int blocks = (elemCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	EncodeKernel<<<blocks, THREADS_PER_BLOCK>>>(
			elements,
			elemCount,
			compressionOutput_device+sizeof(trunkCompressInfo),
			tagBytes,
			metricBytes,
			timeBytes,
			compressedElemSize,
			info);
	cudaDeviceSynchronize();

	// return result
	(*result) = compressionOutput_device;
	return outputSize;
}

size_t DecompressLightweight(unsigned char* data, size_t size, storeElement** result)
{
	// get compression info
	trunkCompressInfo info;
	cudaMemcpy(&info, data, sizeof(trunkCompressInfo), cudaMemcpyDeviceToHost);
	int tagBytes = (info.bytes & 0xFF000000) >> 24;
	int metricBytes = (info.bytes & 0x00FF0000) >> 16;
	int timeBytes = info.bytes & 0x0000FFFF;
	size_t compressedElemSize = tagBytes + metricBytes + timeBytes + 4;
	int elemCount = (size - sizeof(trunkCompressInfo)) / compressedElemSize;

	// prepare decompression output
	storeElement* decompressionOutput_device;	//output space for decompressed data
	cudaMalloc((void**) &decompressionOutput_device, elemCount * sizeof(storeElement));
	cudaMemset(decompressionOutput_device, 0, elemCount * sizeof(storeElement));

	// decompress
	int blocks = (elemCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	unsigned char *decompessionInput_device = data+sizeof(trunkCompressInfo);
	DecodeKernel<<<blocks, THREADS_PER_BLOCK>>>(
			decompessionInput_device,
			elemCount,
			decompressionOutput_device,
			tagBytes,
			metricBytes,
			timeBytes,
			compressedElemSize,
			info);
	cudaDeviceSynchronize();

	// return result
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
	int tag_bytes = (EasyFindLog2(1 + minMax.tag_max - minMax.tag_min) + 7) / 8;
	int metric_bytes = (EasyFindLog2(1 + minMax.metric_max - minMax.metric_min) + 7) / 8;
	int64_t timeDifference = minMax.time_max - minMax.time_min;
	if(timeDifference > std::numeric_limits<int32_t>::max())
		throw std::runtime_error("time difference in trunk is too big!");
	int time_bytes = (EasyFindLog2(1 + (int32_t)(timeDifference)) + 7) / 8;

	// Fill trunk compress info
	trunkCompressInfo info;
	info.tag_min = minMax.tag_min;
	info.metric_min = minMax.metric_min;
	info.time_min = minMax.time_min;
	info.bytes = ((tag_bytes & 0x000000FF) << 24) | ((metric_bytes & 0x000000FF) << 16) | (time_bytes & 0x0000FFFF);

	return info;
}
