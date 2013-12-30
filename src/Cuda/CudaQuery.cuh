#ifndef CUDAQUERY_CUH_
#define CUDAQUERY_CUH_

#include "../Store/StoreElement.h"
#include "../Store/StoreQuery.h"
#include "../Store/StoreTypedefs.h"

// TODO: Move this define to config
#define CUDA_THREADS_PER_BLOCK 256

typedef struct
{
	int32_t tag;
	int metric;
	ullint time;
	float value;
} gpuElem;

extern "C" {

size_t gpu_filterData(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeQuery* query);
size_t gpu_add_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result);

}


#endif /* CUDAQUERY_CUH_ */
