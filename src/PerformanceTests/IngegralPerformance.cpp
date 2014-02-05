#include "IngegralPerformance.h"

namespace ddj {
namespace query {

using namespace std::chrono;
using namespace ddj::store;

	INSTANTIATE_TEST_CASE_P(IngegralPerformanceInst,
					IntegralPerformance,
					::testing::Values(10000000, 20000000, 30000000));

	TEST_P(IntegralPerformance, IntegralAggregation_Sin)
	{
		int X = 50;
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		int numberOfValues = GetParam();
		int trunkSize = bufferSize;
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

			// CLEAN
			delete result;
			delete [] hostData;
			delete dataLocationInfo;
			cudaFree(deviceData);
		}
		printf("[IntegralSin|%d] %d (%f ms)\n", numberOfValues, trunkSize, D.count()/X);
		_resultFile << "IntegralSin"
				<< " " << numberOfValues
				<< " " << trunkSize
				<< " " << D.count()/X << std::endl;
	}

} /* namespace query */
} /* namespace ddj */
