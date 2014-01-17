#ifndef CUDACOMPRESSION_H_
#define CUDACOMPRESSION_H_

#include "../Store/StoreElement.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <limits>

#define THREADS_PER_BLOCK 256
#define COMPRESSED_ELEMENT_SIZE 10
#define COMPRESSED_DATA_SIZE(ELEMENTS_COUNT) (ELEMENTS_COUNT*COMPRESSED_ELEMENT_SIZE+4)

using namespace ddj::store;

struct trunkCompressInfo
{
	int32_t tag_min;
	int32_t metric_min;
	int64_t time_min;
	int32_t bytes;
};

extern "C"
{
	size_t CompressLightweight(storeElement* elements, size_t size, unsigned char** result);
	size_t DecompressLightweight(unsigned char* data, size_t size, storeElement** result);
	trunkCompressInfo AnalizeTrunkData(storeElement* elements, int elemCount);
	void EncodeInt32UsingNBytes(unsigned char* out, int32_t value, int N);
	void EncodeInt64UsingNBytes(unsigned char* out, int64_t value, int N);
}

#endif
