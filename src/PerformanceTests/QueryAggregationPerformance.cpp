/*
 * QueryAggregationPerformance.cpp
 *
 *  Created on: 09-01-2014
 *      Author: ghash
 */

#include "QueryAggregationPerformance.h"

namespace ddj {
namespace query {

INSTANTIATE_TEST_CASE_P(TestOneInst,
						QueryAggregationPerformance,
                        ::testing::Values(100, 10000, 1000000, 5000000, 10000000, 20000000, 30000000));

TEST_P(QueryAggregationPerformance, SumAggregation_EqualIntegerValues)
{
	int N = GetParam();
	size_t dataSize = N*sizeof(storeElement);
	storeElement* deviceData;
	cudaMalloc(&deviceData, dataSize);
	storeElement* hostData = new storeElement[N];
	for(int i=0; i < N; i++) hostData[i].value = 0.01;
	cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

	// EXPECTED
	size_t expected_size = sizeof(results::sumResult);
	float expected_sum = N;
	results::sumResult* result;

	// TEST
	size_t actual_size = _queryAggregation->sum(deviceData, dataSize, (void**)&result, nullptr);

	// CHECK
	ASSERT_EQ(expected_size, actual_size);
	EXPECT_FLOAT_EQ(expected_sum, result->sum);

	// CLEAN
	delete result;
	delete [] hostData;
	cudaFree(deviceData);
}

} /* namespace task */
} /* namespace ddj */
