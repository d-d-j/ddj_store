#include "QueryAggregationPerformance.h"

namespace ddj {
namespace query {

using namespace std::chrono;

INSTANTIATE_TEST_CASE_P(SumEqualIntegersInst,
						QueryAggregationPerformance,
                        ::testing::Values(1, 100, 10000, 1000000, 5000000, 10000000, 20000000));

TEST_P(QueryAggregationPerformance, SumAggregation_EqualIntegerValues)
{
	int N = GetParam();
	int X = 10;
	duration<double, milli> D;
	for(int i=0; i<X; i++)
	{
		size_t dataSize = N*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[N];
		for(int i=0; i < N; i++) hostData[i].value = 0.001;
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(results::sumResult);
		float expected_sum = N*0.001;
		results::sumResult* result;

		// TEST
		auto start = system_clock::now();
		size_t actual_size = _queryAggregation->sum(deviceData, dataSize, (void**)&result, nullptr);
		auto end = system_clock::now();
		D += end - start;

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_NEAR(expected_sum, result->sum, 0.000001*N);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	LOG4CPLUS_INFO(_logger,
			LOG4CPLUS_TEXT("Sum test [N = ")
			<< N
			<< LOG4CPLUS_TEXT("] duration: ")
			<< D.count()/X
			<< LOG4CPLUS_TEXT(" ms"));

	_resultFile << "Sum " << N << " " << D.count()/X << std::endl;
}

} /* namespace task */
} /* namespace ddj */
