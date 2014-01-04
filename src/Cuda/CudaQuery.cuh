#ifndef CUDAQUERY_CUH_
#define CUDAQUERY_CUH_

#include "../Store/StoreElement.cuh"
#include "../Query/AggregationResults.cuh"
#include "../Query/Query.h"
#include "../Store/StoreTypedefs.h"

// TODO: Move this define to config
#define CUDA_THREADS_PER_BLOCK 256

using namespace ddj::store;
using namespace ddj::query;

extern "C" {

	size_t gpu_filterData(storeElement* elements, size_t dataSize, Query* query);

	// AGGREGATION
	size_t gpu_sum(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_max(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_min(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_average(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_stdDeviation(storeElement* elements, size_t dataSize, void** result);

}


#endif /* CUDAQUERY_CUH_ */
