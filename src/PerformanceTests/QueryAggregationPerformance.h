/*
 * QueryAggregationPerformance.h
 *
 *  Created on: 09-01-2014
 *      Author: ghash
 */

#ifndef QUERYAGGREGATIONPERFORMANCE_H_
#define QUERYAGGREGATIONPERFORMANCE_H_

#include "../Query/QueryAggregation.h"
#include "../Query/AggregationResults.cuh"
#include "../Cuda/CudaController.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaIncludes.h"
#include <gtest/gtest.h>
#include <cmath>

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../Core/helper_cuda.h"

namespace ddj {
namespace query {

	class QueryAggregationPerformance : public ::testing::TestWithParam<int>
	{
	protected:
		QueryAggregationPerformance()
		{
			_queryAggregation = new QueryAggregation();
		}
		virtual ~QueryAggregationPerformance()
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
#endif /* QUERYAGGREGATIONPERFORMANCE_H_ */
