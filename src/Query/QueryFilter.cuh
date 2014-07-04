#ifndef QUERYFILTER_CUH_
#define QUERYFILTER_CUH_

#include "QueryAggregationResults.cuh"
#include "../Store/StoreElement.cuh"
#include "../Store/StoreTypedefs.h"
#include "../Core/UllintPair.h"

// TODO: Move this define to config
#define CUDA_THREADS_PER_BLOCK 256

using namespace ddj::store;
using namespace ddj::query;

extern "C" {

	size_t gpu_filterData(
			storeElement* elements,
			size_t dataSize,
			int metricesSize,
			metric_type* metrices,
			int tagsSize,
			int* tags,
			int periodsSize,
			ddj::ullintPair* timePeriods);

	size_t gpu_filterData_in_trunks(
			storeElement* elements,
			size_t dataSize,
			int metricesSize,
			metric_type* metrices,
			int tagsSize,
			int* tags,
			int periodsSize,
			ddj::ullintPair* timePeriods,
			ddj::ullintPair* dataLocationInfo,
			int locationInfoCount);

}


#endif /* QUERYFILTER_CUH_ */
