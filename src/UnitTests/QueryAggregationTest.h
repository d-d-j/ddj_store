#ifndef QUERYAGGREGATIONTEST_H_
#define QUERYAGGREGATIONTEST_H_

#include "../Query/QueryAggregation.h"
#include "../Query/QueryAggregationResults.cuh"
#include "../Cuda/CudaController.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaIncludes.h"
#include <gtest/gtest.h>
#include <cmath>

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../Core/helper_cuda.h"
#include <thrust/version.h>

namespace ddj {
namespace query {

using namespace store;

	class QueryAggregationTest : public ::testing::Test
	{
	protected:
		QueryAggregationTest()
		{
			_queryAggregation = new QueryAggregation();
		}
		~QueryAggregationTest()
		{
			delete _queryAggregation;
		}
		virtual void SetUp()
		{
			const char* argv = "";
			cudaSetDevice(findCudaDevice(0, &argv));
		}

		QueryAggregation* _queryAggregation;
	};

} /* namespace query */
} /* namespace ddj */


#endif /* QUERYAGGREGATIONTEST_H_ */
