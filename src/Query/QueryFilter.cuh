#ifndef QUERYFILTER_CUH_
#define QUERYFILTER_CUH_

#include "QueryAggregationResults.cuh"
#include "Query.h"
#include "../Store/StoreElement.cuh"
#include "../Store/StoreTypedefs.h"

// TODO: Move this define to config
#define CUDA_THREADS_PER_BLOCK 256

using namespace ddj::store;
using namespace ddj::query;

extern "C" {

	size_t gpu_filterData(storeElement* elements, size_t dataSize, Query* query);
	size_t gpu_filterData_in_trunks(storeElement* elements, size_t dataSize, Query* query,
				ddj::ullintPair* dataLocationInfo, int locationInfoCount);

}


#endif /* QUERYFILTER_CUH_ */
