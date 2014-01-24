#include "QueryAggregationPerformance.h"
#include <cmath>

namespace ddj {
namespace query {

using namespace std::chrono;

	INSTANTIATE_TEST_CASE_P(QueryAggregationInst,
						QueryAggregationPerformance,
						::testing::Values(1, 200, 2000, 20000, 200000, 2000000, 20000000));

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
//			EXPECT_NEAR(expected_sum, result->sum, 0.000001*N);

			// CLEAN
			delete result;
			delete [] hostData;
			cudaFree(deviceData);
		}

//		LOG4CPLUS_INFO(_logger,
//				LOG4CPLUS_TEXT("Sum test [N = ")
//				<< N
//				<< LOG4CPLUS_TEXT("] duration: ")
//				<< D.count()/X
//				<< LOG4CPLUS_TEXT(" ms"));
		printf("[SumEqualIntegers|%d] (%f ms)\n", N, D.count()/X);
		_resultFile << "Sum " << N << " " << D.count()/X << std::endl;
	}

	TEST_P(QueryAggregationPerformance, IntegralAggregation_Sin_Trunk200)
	{
		int X = 10;
		duration<double, milli> D;
		int numberOfValues = GetParam();	// 40 elements with 4 trunks (10,10,10,10) - number of elements in each trunk
		int trunkSize = 200;
		int trunkCount = (numberOfValues + trunkSize - 1) / trunkSize;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		for(int k=0; k<X; k++)
		{
			storeElement* hostData = new storeElement[numberOfValues];
			for(int i=0; i< numberOfValues; i++)
			{
				hostData[i].value = std::sin(i/100.0f*M_PI);
				hostData[i].time = 2*i;
			}

			// COPY TO DEVICE
			storeElement* deviceData;
			cudaMalloc(&deviceData, dataSize);
			cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

			// DATA LOCATION INFO
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			size_t oneTrunkSize = trunkSize*sizeof(storeElement);
			size_t lastTrunkSize = (numberOfValues%oneTrunkSize)*sizeof(storeElement);
			lastTrunkSize = lastTrunkSize ? lastTrunkSize : oneTrunkSize;
			for(int i=0; i<trunkCount; i++)
			{
				if(i != trunkCount - 1)
					dataLocationInfo->push_back(ullintPair{i*oneTrunkSize,(i+1)*oneTrunkSize-1});
				else
					dataLocationInfo->push_back(ullintPair{i*oneTrunkSize,i*oneTrunkSize+lastTrunkSize-1});
			}

			// QUERY
			Query query;
			query.aggregationData = dataLocationInfo;

			// EXPECTED
			size_t expected_size = trunkCount*sizeof(results::integralResult);
			float expected_integral = 0.0f;
			float eps = 0.1f ;
			results::integralResult* result;

			// TEST
			auto start = system_clock::now();
			size_t actual_size =
					_queryAggregation->_aggregationFunctions[AggregationType::Integral](deviceData, dataSize, (void**)&result, &query);
			auto end = system_clock::now();
			D += end - start;

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
//			for(int j=0; j<trunkCount; j++)
//			{
//				if(numberOfValues%oneTrunkSize == 0 || j < trunkCount-1 )
//					EXPECT_NEAR(expected_integral, result[j].integral, eps);
//			}

			// CLEAN
			delete result;
			delete [] hostData;
			delete dataLocationInfo;
			cudaFree(deviceData);
		}
//		LOG4CPLUS_INFO(_logger,
//						LOG4CPLUS_TEXT("Integral SinTrunk200 test [N = ")
//						<< numberOfValues
//						<< LOG4CPLUS_TEXT("] duration: ")
//						<< D.count()/X
//						<< LOG4CPLUS_TEXT(" ms"));
		printf("[SinTrunk200|%d] (%f ms)\n", numberOfValues, D.count()/X);
		_resultFile << "Integral_SinTrunk200 " << numberOfValues << " " << D.count()/X << std::endl;
	}

} /* namespace task */
} /* namespace ddj */
