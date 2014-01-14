#ifndef CUDACOMPRESSION_H_
#define CUDACOMPRESSION_H_

#include "../Store/StoreElement.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BLOCKS 50
#define THREADS 100
#define ELEMENTS_COUNT (BLOCKS*THREADS)
#define DATA_SIZE (ELEMENTS_COUNT * sizeof(storeElement))
#define COMPRESSED_ELEMENT_SIZE 10
#define COMPRESSED_DATA_SIZE (ELEMENTS_COUNT*COMPRESSED_ELEMENT_SIZE+4)

using namespace ddj::store;

extern "C"
{
	size_t CompressLightweight(storeElement* elements, size_t size, unsigned char** result);
	size_t DecompressLightweight(unsigned char* data, size_t size, storeElement** result);
}

#endif
