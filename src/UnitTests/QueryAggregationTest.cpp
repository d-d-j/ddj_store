#include "QueryAggregationTest.h"

namespace ddj {
namespace query {

/***********************/
/* AGGREGATION MATHODS */
/***********************/

//sum

	TEST_F(QueryAggregationTest, sum_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size = _queryAggregation->sum(elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, sum_OneElement)
	{
		// PREPARE
		int numberOfValues = 1;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		hostData[0].value = 3.7425f;
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(results::sumResult);
		float expected_sum = 3.7425f;
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

	TEST_F(QueryAggregationTest, sum_EvenNumberOfValues)
	{
		// PREPARE
		int numberOfValues = 123;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) hostData[i].value = 3;
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(results::sumResult);
		float expected_sum = 3*123;
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

	TEST_F(QueryAggregationTest, sum_OddNumberOfValues)
	{
		// PREPARE
		int numberOfValues = 2000;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) hostData[i].value = 4.2f;
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(results::sumResult);
		float expected_sum = 4.2f*2000;
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

//max

	TEST_F(QueryAggregationTest, max_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size = _queryAggregation->max(elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, max_Positive)
	{
		// PREPARE
		int numberOfValues = 2000;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			hostData[i].value =-(i*i) + i*180.0f;	// max for i = 90 is 90*90 = 8100
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(storeElement);
		float expected_max = 8100.0f;
		storeElement* result;

		// TEST
		size_t actual_size = _queryAggregation->max(deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_max, result->value);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, max_Negative)
	{
		// PREPARE
		int numberOfValues = 2000;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			hostData[i].value =(i*i) - i*180.0f;	// max for i = 1999 is 1999*1999-1999*180 = 3636181
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(storeElement);
		float expected_max = 3636181.0f;
		storeElement* result;

		// TEST
		size_t actual_size = _queryAggregation->max(deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_max, result->value);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

//min

	TEST_F(QueryAggregationTest, min_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size = _queryAggregation->min(elements, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, min_Positive)
	{
		// PREPARE
		int numberOfValues = 2000;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			hostData[i].value =(i*i)+3.0f;	// min for i=1000 is 3
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(storeElement);
		float expected_min = 3.0f;
		storeElement* result;

		// TEST
		size_t actual_size = _queryAggregation->min(deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_min, result->value);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, min_Negative)
	{
		// PREPARE
		int numberOfValues = 2000;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			hostData[i].value =(i*i) - i*180.0f;	// min for i = 90 is -90*90 = -8100
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(storeElement);
		float expected_min = -8100.0f;
		storeElement* result;

		// TEST
		size_t actual_size = _queryAggregation->min(deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_min, result->value);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

//average

	TEST_F(QueryAggregationTest, average_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size = _queryAggregation->_aggregationFunctions[AggregationType::Average](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, average_Linear)
	{
		// PREPARE
		int numberOfValues = 2001;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			int x = i - 1000;
			hostData[i].value = x;	// average = 0
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(results::averageResult);
		float expected_sum = 0.0f;
		int32_t expected_count = 2001;
		results::averageResult* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Average](deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_sum, result->sum);
		EXPECT_EQ(expected_count, result->count);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, average_Sinusoidal)
	{
		// PREPARE
		int numberOfValues = 2001;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			hostData[i].value = std::sin(i*M_PI/4.0f);
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		size_t expected_size = sizeof(results::averageResult);
		float expected_sum = 0.0f;
		int expected_count = 2001;
		results::averageResult* result;

		// TEST
		size_t actual_size
		= _queryAggregation->_aggregationFunctions[AggregationType::Average](deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_sum, result->sum);
		EXPECT_EQ(expected_count, result->count);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

//stdDeviation or Variance

	TEST_F(QueryAggregationTest, stdDeviationOrVariance_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::StdDeviation](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);

		// TEST
		actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Variance](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, stdDeviationOrVariance_Simple)
	{
		// PREPARE
		int numberOfValues = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		hostData[0].value = 5;
		hostData[1].value = 6;
		hostData[2].value = 8;
		hostData[3].value = 9;
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// EXPECTED
		/*
		 * number of values: n = 4
		 * values: {5, 6, 8, 9}
		 * average: a = 7
		 * standard deviation: s = sqrt[1/(n-1) * SUM[i=0 to 3: (values[i] - a)^2] ]
		 * s = sqrt[1/3 * (4+1+1+4)] = sqrt[10/3]
		 */
		size_t expected_size = sizeof(results::varianceResult);
		int expected_count = 4;
		float expected_mean = 7.0f;
		float expected_M2 = 10.0f;
		results::varianceResult* result;

		// TEST
		size_t actual_size
			= _queryAggregation->_aggregationFunctions[AggregationType::StdDeviation](deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(expected_count, result->count);
		EXPECT_FLOAT_EQ(expected_mean, result->mean);
		EXPECT_FLOAT_EQ(expected_M2, result->m2);

		// CLEAN
		delete result;

		// TEST
		actual_size
			= _queryAggregation->_aggregationFunctions[AggregationType::Variance](deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(expected_count, result->count);
		EXPECT_FLOAT_EQ(expected_mean, result->mean);
		EXPECT_FLOAT_EQ(expected_M2, result->m2);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, stdDeviationOrVariance_Linear)
	{
		// PREPARE
		int numberOfValues = 2001;
		size_t dataSize = numberOfValues*sizeof(storeElement);
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i < numberOfValues; i++) {
			hostData[i].value = i;	// 0,1,2,3,...,2000
		}
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
		storeElement hostResult;

		// EXPECTED
		/*
		 * number of values: n = 2001
		 * values: {0, 1,..., 1999, 2000}
		 * average: a = 1000
		 * standard deviation: s = sqrt[1/2000 * SUM[i=0 to 2000: (values[i] - 1000)^2] ]
		 * s = sqrt[667667/2]
		 */
		size_t expected_size = sizeof(results::varianceResult);
		int expected_count = 2001;
		float expected_mean = 1000.0f;
		float expected_M2 = 667667000.0f;
		results::varianceResult* result;

		// TEST
		size_t actual_size
			= _queryAggregation->_aggregationFunctions[AggregationType::StdDeviation](deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(expected_count, result->count);
		EXPECT_FLOAT_EQ(expected_mean, result->mean);
		EXPECT_FLOAT_EQ(expected_M2, result->m2);

		// CLEAN
		delete result;

		// TEST
		actual_size
			= _queryAggregation->_aggregationFunctions[AggregationType::Variance](deviceData, dataSize, (void**)&result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(expected_count, result->count);
		EXPECT_FLOAT_EQ(expected_mean, result->mean);
		EXPECT_FLOAT_EQ(expected_M2, result->m2);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

//integral

	TEST_F(QueryAggregationTest, integral_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Integral](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, integral_Simple_OneTrunk)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		int numberOfValues = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		hostData[0].value = 2.0f; hostData[0].time = 10; // (10,2)
		hostData[1].value = 4.0f; hostData[1].time = 20; // (20,4)
		hostData[2].value = 4.0f; hostData[2].time = 30; // (30,4)
		hostData[3].value = 2.0f; hostData[3].time = 40; // (40,2)

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// DATA LOCATION INFO
		boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
		dataLocationInfo->push_back(ullintPair{0,numberOfValues*sizeof(storeElement)-1});

		// QUERY
		Query query;
		query.aggregationData = dataLocationInfo;

		// EXPECTED
		size_t expected_size = dataLocationInfo->size()*sizeof(results::integralResult);
		float expected_integral = 100.0f;
		float expected_left_value = 2.0f;
		float expected_right_value = 2.0f;
		int64_t expected_left_time = 10;
		int64_t expected_right_time = 40;
		results::integralResult* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Integral](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_integral, result->integral);
		EXPECT_FLOAT_EQ(expected_left_value, result->left_value);
		EXPECT_FLOAT_EQ(expected_right_value, result->right_value);
		EXPECT_EQ(expected_left_time, result->left_time);
		EXPECT_EQ(expected_right_time, result->right_time);

		// CLEAN
		delete result;
		delete [] hostData;
		delete dataLocationInfo;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, integral_Simple_OneTrunk_SingleElement)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		int numberOfValues = 1;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		hostData[0].value = 2.0f; hostData[0].time = 10; // (10,2)

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// DATA LOCATION INFO
		boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
		dataLocationInfo->push_back(ullintPair{0,numberOfValues*sizeof(storeElement)-1});

		// QUERY
		Query query;
		query.aggregationData = dataLocationInfo;

		// EXPECTED
		size_t expected_size = dataLocationInfo->size()*sizeof(results::integralResult);
		float expected_integral = 0.0f;
		float expected_left_value = 2.0f;
		float expected_right_value = 2.0f;
		int64_t expected_left_time = 10;
		int64_t expected_right_time = 10;
		results::integralResult* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Integral](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_FLOAT_EQ(expected_integral, result->integral);
		EXPECT_FLOAT_EQ(expected_left_value, result->left_value);
		EXPECT_FLOAT_EQ(expected_right_value, result->right_value);
		EXPECT_EQ(expected_left_time, result->left_time);
		EXPECT_EQ(expected_right_time, result->right_time);

		// CLEAN
		delete result;
		delete [] hostData;
		delete dataLocationInfo;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, integral_Simple_ManyTrunks_EqualTrunks)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		int numberOfValues = 40;	// 40 elements with 4 trunks (10,10,10,10) - number of elements in each trunk
		int numberOfTrunks = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i< numberOfValues; i++)
		{
			hostData[i].value = (i%2)+1; hostData[i].time = 2*i; // (1,0),(2,2),(1,4),(2,6),...
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// DATA LOCATION INFO
		boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
		size_t oneTrunkSize = (numberOfValues/numberOfTrunks)*sizeof(storeElement);
		dataLocationInfo->push_back(ullintPair{0*oneTrunkSize,1*oneTrunkSize-1});
		dataLocationInfo->push_back(ullintPair{1*oneTrunkSize,2*oneTrunkSize-1});
		dataLocationInfo->push_back(ullintPair{2*oneTrunkSize,3*oneTrunkSize-1});
		dataLocationInfo->push_back(ullintPair{3*oneTrunkSize,4*oneTrunkSize-1});

		// QUERY
		Query query;
		query.aggregationData = dataLocationInfo;

		// EXPECTED
		size_t expected_size = numberOfTrunks*sizeof(results::integralResult);
		float expected_integral = 27.0f;
		float expected_left_value = 1.0f;
		float expected_right_value = 2.0f;
		results::integralResult* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Integral](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfTrunks; j++)
		{
			EXPECT_FLOAT_EQ(expected_integral, result[j].integral);
			EXPECT_FLOAT_EQ(expected_left_value, result[j].left_value);
			EXPECT_FLOAT_EQ(expected_right_value, result[j].right_value);
			EXPECT_EQ(j*20, result[j].left_time);
			EXPECT_EQ(j*20+18, result[j].right_time);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		delete dataLocationInfo;
		cudaFree(deviceData);
	}

//histogram on values

	TEST_F(QueryAggregationTest, histogram_Value_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Value](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, histogram_Value_Simple_1Bucket)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 500;
		const int numberOfBuckets = 1;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			hostData[i].value = i;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramValueData data(100.0f, 300.0f, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expected_result = 200;
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Value](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(expected_result, result[0]);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, histogram_Value_Simple_4Buckets)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 100;
		const int numberOfBuckets = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			if(i < 20) hostData[i].value = -8.0f;
			if(i >= 20 && i < 50) hostData[i].value = -2.0f;
			if(i >= 50 && i < 80) hostData[i].value = 2.0f;
			if(i >= 80)hostData[i].value = 8.0f;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramValueData data(-10.0f, 10.0f, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expectedResults[numberOfBuckets] = {20,30,30,20};
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Value](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfBuckets; j++)
		{
			EXPECT_EQ(expectedResults[j], result[j]);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, histogram_Value_ValuesOnBucketsEdges_LeftInclusive_4Buckets)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 100;
		const int numberOfBuckets = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			if(i < 20) hostData[i].value = -10.0f;
			if(i >= 20 && i < 50) hostData[i].value = -5.0f;
			if(i >= 50 && i < 80) hostData[i].value = 0.0f;
			if(i >= 80)hostData[i].value = 5.0f;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramValueData data(-10.0f, 10.0f, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expectedResults[numberOfBuckets] = {20,30,30,20};
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Value](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfBuckets; j++)
		{
			EXPECT_EQ(expectedResults[j], result[j]);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, histogram_Value_ValuesOnBucketsEdges_RightExclusive_4Buckets)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 100;
		const int numberOfBuckets = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			if(i < 20) hostData[i].value = -5.0f;
			if(i >= 20 && i < 50) hostData[i].value = 0.0f;
			if(i >= 50 && i < 80) hostData[i].value = 5.0f;
			if(i >= 80)hostData[i].value = 10.0f;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramValueData data(-10.0f, 10.0f, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expectedResults[numberOfBuckets] = {0,20,30,30};
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Value](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfBuckets; j++)
		{
			EXPECT_EQ(expectedResults[j], result[j]);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

//histogram on time

	TEST_F(QueryAggregationTest, histogram_Time_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Time](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, histogram_Time_Simple_1Bucket)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 500;
		const int numberOfBuckets = 1;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			hostData[i].time = i;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramTimeData data(100, 300, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;
		query.timePeriods.push_back({100,300});

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expected_result = 200;
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Time](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(expected_result, result[0]);

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, histogram_Time_Simple_4Buckets)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 100;
		const int numberOfBuckets = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			if(i < 20) hostData[i].time = 2;
			if(i >= 20 && i < 50) hostData[i].time = 8;
			if(i >= 50 && i < 80) hostData[i].time = 12;
			if(i >= 80)hostData[i].time = 17;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramTimeData data(0, 20, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expectedResults[numberOfBuckets] = {20,30,30,20};
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Time](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfBuckets; j++)
		{
			EXPECT_EQ(expectedResults[j], result[j]);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, histogram_Time_ValuesOnBucketsEdges_LeftInclusive_4Buckets)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 100;
		const int numberOfBuckets = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			if(i < 20) hostData[i].time = 0;
			if(i >= 20 && i < 50) hostData[i].time = 5;
			if(i >= 50 && i < 80) hostData[i].time = 10;
			if(i >= 80)hostData[i].time = 15;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramTimeData data(0, 20, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expectedResults[numberOfBuckets] = {20,30,30,20};
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Time](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfBuckets; j++)
		{
			EXPECT_EQ(expectedResults[j], result[j]);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, histogram_Time_ValuesOnBucketsEdges_RightExclusive_4Buckets)
	{
		/////////////////
		//// PREPARE ////
		/////////////////
		const int numberOfValues = 100;
		const int numberOfBuckets = 4;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			if(i < 20) hostData[i].time = 5;
			if(i >= 20 && i < 50) hostData[i].time = 10;
			if(i >= 50 && i < 80) hostData[i].time = 15;
			if(i >= 80)hostData[i].time = 20;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		// CREATE HISTOGRAM DATA
		data::histogramTimeData data(0, 20, numberOfBuckets);

		// QUERY
		Query query;
		query.aggregationData = &data;

		// EXPECTED
		size_t expected_size = numberOfBuckets*sizeof(int);
		int expectedResults[numberOfBuckets] = {0,20,30,30};
		int* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::Histogram_Time](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfBuckets; j++)
		{
			EXPECT_EQ(expectedResults[j], result[j]);
		}

		// CLEAN
		delete result;
		delete [] hostData;
		cudaFree(deviceData);
	}

//sum series

	TEST_F(QueryAggregationTest, series_Sum_Empty)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](elements, dataSize, &result, nullptr);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, series_Sum_WrongQuery_NoTimePeriods)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;
		Query query;
		query.metrics.push_back(0);
		query.tags.push_back(0);
		query.aggregationData = new data::interpolatedAggregationData(4);

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](elements, dataSize, &result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, series_Sum_WrongQuery_NoTags)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;
		Query query;
		query.metrics.push_back(0);
		query.timePeriods.push_back({1,12});
		query.aggregationData = new data::interpolatedAggregationData(4);

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](elements, dataSize, &result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, series_Sum_WrongQuery_NoMetrics)
	{
		// PREPARE
		storeElement* elements = nullptr;
		size_t dataSize = 0;
		void* result;
		Query query;
		query.tags.push_back(0);
		query.timePeriods.push_back({1,12});
		query.aggregationData = new data::interpolatedAggregationData(4);

		// EXPECTED
		size_t expected_size = 0;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](elements, dataSize, &result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		EXPECT_EQ(nullptr, result);
	}

	TEST_F(QueryAggregationTest, series_Sum_Simple_3tags1metric_EqualValues_ConsistentTimeIntervals)
	{
		// PREPARE
		const int numberOfSeries = 3;
		const int numberOfTimePoints = 4;
		const int numberOfValuesInOneSeries = 12;
		const int numberOfValues = numberOfSeries*numberOfValuesInOneSeries;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			hostData[i].metric = 0;
			hostData[i].value = 3.0f;
			hostData[i].time = (i%numberOfValuesInOneSeries) + 1;
			if(i < 12) hostData[i].tag = 0;
			if(i >= 12 && i < 24) hostData[i].tag = 11;
			if(i >= 24) hostData[i].tag = 333;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		Query query;
		query.tags.push_back(0);
		query.tags.push_back(11);
		query.tags.push_back(333);
		query.metrics.push_back(0);
		// 1,4,7,10 - time points
		query.timePeriods.push_back({1,numberOfValuesInOneSeries});
		query.aggregationData = new data::interpolatedAggregationData(numberOfTimePoints);

		// EXPECTED
		size_t expected_size = numberOfTimePoints*sizeof(float);
		float expected_results[numberOfTimePoints] = {9.0f, 9.0f, 9.0f, 9.0f};
		float* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfTimePoints; j++)
		{
			EXPECT_EQ(expected_results[j], result[j]);
		}

		// CLEAN
		delete [] result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, series_Sum_Simple_3tags3metrics_LinearValues_ConsistentTimeIntervals)
	{
		// PREPARE
		const int numberOfMetrics = 3;
		const int numberOfTags = 3;
		const int numberOfTimePoints = 4;
		const int numberOfValuesInOneSeries = 12;
		const int numberOfSeries = numberOfMetrics*numberOfTags;
		const int numberOfValues = numberOfSeries*numberOfValuesInOneSeries;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValuesInOneSeries; i++)
		{
			for(int j=0; j<numberOfMetrics; j++)
			{
				for(int k=0; k<numberOfTags; k++)
				{
					hostData[i*9+j*3+k].metric = j;
					hostData[i*9+j*3+k].tag = k;
					hostData[i*9+j*3+k].value = (i+1)*1.0f;
					hostData[i*9+j*3+k].time = i+1;
				}
			}
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		Query query;
		query.tags.push_back(0);
		query.tags.push_back(1);
		query.tags.push_back(2);
		query.metrics.push_back(0);
		query.metrics.push_back(1);
		query.metrics.push_back(2);
		// 1,5,9,12 - time points
		query.timePeriods.push_back({1,numberOfValuesInOneSeries});
		query.aggregationData = new data::interpolatedAggregationData(numberOfTimePoints);

		// EXPECTED
		size_t expected_size = numberOfTimePoints*sizeof(float);
		float expected_results[numberOfTimePoints] = {9.0f, 36.0f, 63.0f, 90.0f};
		float* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfTimePoints; j++)
		{
			EXPECT_EQ(expected_results[j], result[j]);
		}

		// CLEAN
		delete [] result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, series_Sum_Simple_3tags1metric_LinearValues_InterpolationNeeded)
	{
		// PREPARE
		const int numberOfSeries = 3;
		const int numberOfTimePoints = 3;
		const int numberOfValuesInOneSeries = 4;
		const int numberOfValues = numberOfSeries*numberOfValuesInOneSeries;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValues; i++)
		{
			hostData[i].metric = 0;
			hostData[i].value = 2*(i/numberOfSeries)+1;
			hostData[i].time = 2*(i/numberOfSeries)+1;
			hostData[i].tag = i%numberOfSeries;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		Query query;
		query.tags.push_back(0);
		query.tags.push_back(1);
		query.tags.push_back(2);
		query.metrics.push_back(0);
		// 2,4,6 - time points
		query.timePeriods.push_back({2,7});
		query.aggregationData = new data::interpolatedAggregationData(numberOfTimePoints);

		// EXPECTED
		size_t expected_size = numberOfTimePoints*sizeof(float);
		float expected_results[numberOfTimePoints] = {6.0f, 12.0f, 18.0f};
		float* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfTimePoints; j++)
		{
			EXPECT_EQ(expected_results[j], result[j]);
		}

		// CLEAN
		delete [] result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, series_Sum_Normal_2tags1metrics_SinCosValues_ConsistentTimeIntervals)
	{
		// PREPARE
		const int numberOfMetrics = 1;
		const int numberOfTags = 2;
		const int numberOfTimePoints = 400;
		const int numberOfValuesInOneSeries = 1200;
		const int numberOfSeries = numberOfMetrics*numberOfTags;
		const int numberOfValues = numberOfSeries*numberOfValuesInOneSeries;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValuesInOneSeries; i++)
		{
			if(i%2)
			{
				hostData[i].value = std::pow(std::sin((i*1.0f)/100.0f*M_PI),2);
				hostData[i].tag = 11;
			}
			else
			{
				hostData[i].value = std::pow(std::cos((i*1.0f)/100.0f*M_PI),2);
				hostData[i].tag = 99;
			}
			hostData[i].metric = 0;
			hostData[i].time = i;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		Query query;
		query.tags.push_back(11);
		query.tags.push_back(99);
		query.metrics.push_back(0);
		// 1,5,9,12 - time points
		query.timePeriods.push_back({111,999});
		query.aggregationData = new data::interpolatedAggregationData(numberOfTimePoints);

		// EXPECTED
		size_t expected_size = numberOfTimePoints*sizeof(float);
		float expected_result = 1.0f;
		float eps_error = 0.001f;
		float* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfTimePoints; j++)
		{
			EXPECT_NEAR(expected_result, result[j], eps_error);
		}

		// CLEAN
		delete [] result;
		delete [] hostData;
		cudaFree(deviceData);
	}

	TEST_F(QueryAggregationTest, series_Sum_Normal_2tags1metrics_SinCosValues_InterpolationNeeded)
	{
		// PREPARE
		const int numberOfMetrics = 1;
		const int numberOfTags = 2;
		const int numberOfTimePoints = 400;
		const int numberOfValuesInOneSeries = 1200;
		const int numberOfSeries = numberOfMetrics*numberOfTags;
		const int numberOfValues = numberOfSeries*numberOfValuesInOneSeries;
		size_t dataSize = numberOfValues*sizeof(storeElement);

		storeElement* hostData = new storeElement[numberOfValues];
		for(int i=0; i<numberOfValuesInOneSeries; i++)
		{
			if(i%2)
			{
				hostData[i].value = std::pow(std::sin((i*1.0f)/100.0f*M_PI),2);
				hostData[i].tag = 11;
			}
			else
			{
				hostData[i].value = std::pow(std::cos((i*1.0f)/100.0f*M_PI),2);
				hostData[i].tag = 99;
			}
			hostData[i].metric = 0;
			hostData[i].time = 2*i;
		}

		// COPY TO DEVICE
		storeElement* deviceData;
		cudaMalloc(&deviceData, dataSize);
		cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

		Query query;
		query.tags.push_back(11);
		query.tags.push_back(99);
		query.metrics.push_back(0);
		// 1,5,9,12 - time points
		query.timePeriods.push_back({200,1400});
		query.aggregationData = new data::interpolatedAggregationData(numberOfTimePoints);

		// EXPECTED
		size_t expected_size = numberOfTimePoints*sizeof(float);
		float expected_result = 1.0f;
		float eps_error = 0.001f;
		float* result;

		// TEST
		size_t actual_size =
				_queryAggregation->_aggregationFunctions[AggregationType::SumSeries](deviceData, dataSize, (void**)&result, &query);

		// CHECK
		ASSERT_EQ(expected_size, actual_size);
		for(int j=0; j<numberOfTimePoints; j++)
		{
			EXPECT_NEAR(expected_result, result[j], eps_error);
		}

		// CLEAN
		delete [] result;
		delete [] hostData;
		cudaFree(deviceData);
	}

} /* namespace query */
} /* namespace ddj */
