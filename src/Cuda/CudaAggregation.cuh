#ifndef CUDAAGGREGATION_CUH_
#define CUDAAGGREGATION_CUH_

#include "../Store/StoreElement.cuh"
#include "../Query/AggregationResults.cuh"
#include "../Query/AggregationData.h"
#include "../Query/Query.h"
#include "../Store/StoreTypedefs.h"
#include <vector_types.h>

// TODO: Move this define to config
#define CUDA_THREADS_PER_BLOCK 256

using namespace ddj::store;
using namespace ddj::query;

extern "C" {

	size_t gpu_sum(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_max(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_min(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_average(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_variance(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_skewness(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_kurtosis(storeElement* elements, size_t dataSize, void** result);
	size_t gpu_histogram_value(storeElement* elements, size_t dataSize, void** result, float2* buckets, int bucketCount);
	size_t gpu_histogram_time(storeElement* elements, size_t dataSize, void** result, ullint2* buckets, int bucketCount);
	size_t gpu_trunk_integral(storeElement* elements, size_t dataSize, void** result,
			ddj::ullintPair* dataLocationInfo, int locationInfoCount);

}

#endif /* CUDAAGGREGATION_CUH_ */
