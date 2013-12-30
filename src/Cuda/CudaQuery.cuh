#ifndef CUDAQUERY_CUH_
#define CUDAQUERY_CUH_

#include "../Store/StoreElement.h"
#include "../Store/StoreQuery.h"
#include "../Store/StoreTypedefs.h"

// TODO: Move this define to config
#define CUDA_THREADS_PER_BLOCK 256

// TODO: Move StoreElement to .cuh file and remove gpuElem to use only StoreElement everywhere
struct gpuElem
{
	int32_t tag;
	int metric;
	ullint time;
	float value;

	__host__ __device__
	gpuElem()
	{
		tag = 0;
		metric = 0;
		time = 0;
		value = 0;
	}

	gpuElem& operator= (const ddj::store::storeElement& elem)
	{
		tag = elem.tag;
		metric = elem.metric;
		time = elem.time;
		value = elem.value;
		return *this;
	}

	gpuElem (const ddj::store::storeElement& elem)
	{
		tag = elem.tag;
		metric = elem.metric;
		time = elem.time;
		value = elem.value;
	}
};

extern "C" {

size_t gpu_filterData(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeQuery* query);
size_t gpu_add_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result);
size_t gpu_max_from_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result);
size_t gpu_min_from_values(ddj::store::storeElement* elements, size_t dataSize, ddj::store::storeElement** result);

}


#endif /* CUDAQUERY_CUH_ */
