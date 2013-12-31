/*
 * StoreQueryCoreTest.cpp
 *
 *  Created on: 17-12-2013
 *      Author: ghash
 */

#include "StoreQueryCoreTest.h"
#include <thrust/version.h>
#include "../Cuda/CudaIncludes.h"
#define STORE_QUERY_CORE_TEST_MEM_SIZE 1024

namespace ddj {
namespace store {

	void StoreQueryCoreTest::createSimpleCharTestData()
	{
		void* mainMemoryPointer = _cudaController->GetMainMemoryPointer();
		size_t size = STORE_QUERY_CORE_TEST_MEM_SIZE;
		char* testArray = new char[size];
		// FILL TAST ARRAY WITH CHARACTERS
		for(unsigned long i=0; i<size; i++)
			testArray[i] = (char)(i%256);

		CUDA_CHECK_RETURN( cudaMemcpy(mainMemoryPointer, testArray, size, cudaMemcpyHostToDevice) );
		_cudaController->SetMainMemoryOffset(size);
	}

	void StoreQueryCoreTest::createTestDataWithStoreElements()
	{
		void* mainMemoryPointer = _cudaController->GetMainMemoryPointer();
		size_t size = STORE_QUERY_CORE_TEST_MEM_SIZE;
		storeElement* testArray = new storeElement[size];

		// FILL TAST ARRAY WITH STORE ELEMENTS
		for(unsigned long i=0; i<size; i++)
		{
			testArray[i].tag = i%20;
			testArray[i].metric = 1;
			testArray[i].time = i*10;
			testArray[i].value = 3;
		}

		CUDA_CHECK_RETURN( cudaMemcpy(mainMemoryPointer, testArray, size*sizeof(storeElement), cudaMemcpyHostToDevice) );
		_cudaController->SetMainMemoryOffset(size*sizeof(storeElement));
	}

	// check thrust version
	TEST_F(StoreQueryCoreTest, ThrustVersion)
	{
		int major = THRUST_MAJOR_VERSION;
		int minor = THRUST_MINOR_VERSION;
		RecordProperty("Thrust version major", major);
		RecordProperty("Thrust version minor", minor);
		EXPECT_EQ(1, major);
		EXPECT_EQ(7, minor);
	}

	/***************************/
	/* DATA MANAGEMENT METHODS */
	/***************************/

	//mapData

		TEST_F(StoreQueryCoreTest, mapData_AllData)
		{
			// PREPARE
			createSimpleCharTestData();
			char* deviceData;
			char* hostData;

			// TEST
			size_t size = _queryCore->mapData((void**)&deviceData);

			// CHECK
			ASSERT_EQ(STORE_QUERY_CORE_TEST_MEM_SIZE, size);

			hostData = new char[size];
			CUDA_CHECK_RETURN( cudaMemcpy(hostData, deviceData, size, cudaMemcpyDeviceToHost) );

			for(unsigned long i=0; i<size; i++)
				EXPECT_EQ((char)(i%256), hostData[i]);

			// CLEAN
			delete [] hostData;
			CUDA_CHECK_RETURN( cudaFree(deviceData) );
		}

		TEST_F(StoreQueryCoreTest, mapData_ChooseOneTrunk)
		{
			// PREPARE
			createSimpleCharTestData();
			char* deviceData;
			char* hostData;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{64,127});

			// TEST
			size_t size = _queryCore->mapData((void**)&deviceData, dataLocationInfo);

			// CHECK
			ASSERT_EQ(64, size);

			hostData = new char[size];
			CUDA_CHECK_RETURN( cudaMemcpy(hostData, deviceData, size, cudaMemcpyDeviceToHost) );

			for(unsigned long i=0; i<size; i++)
				EXPECT_EQ((char)((i+size)%256), hostData[i]);

			// CLEAN
			delete [] hostData;
			CUDA_CHECK_RETURN( cudaFree(deviceData) );
		}

		TEST_F(StoreQueryCoreTest, mapData_ChooseManyTrunks)
		{
			// PREPARE
			createSimpleCharTestData();
			char* deviceData;
			char* hostData;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{64,127});	// length 64
			dataLocationInfo->push_back(ullintPair{256,383});	// length 128
			dataLocationInfo->push_back(ullintPair{601,728});	// length 128

			// TEST
			size_t size = _queryCore->mapData((void**)&deviceData, dataLocationInfo);

			// CHECK
			ASSERT_EQ(64+128+128, size);

			hostData = new char[size];
			CUDA_CHECK_RETURN( cudaMemcpy(hostData, deviceData, size, cudaMemcpyDeviceToHost) );



			unsigned long i = 0;
			for(i=0; i<64; i++)
				EXPECT_EQ((char)((i+64)%256), hostData[i]);

			for(i=64; i<64+128; i++)
				EXPECT_EQ((char)((i+256-64)%256), (char)hostData[i]);

			for(i=64+128; i<64+128+128; i++)
				EXPECT_EQ((char)((i+601-64-128)%256), (char)hostData[i]);

			// CLEAN
			delete [] hostData;
			CUDA_CHECK_RETURN( cudaFree(deviceData) );
		}

	//filterData

		TEST_F(StoreQueryCoreTest, filterData_AllData)
		{
			// PREPARE
			int N = 100;
			storeElement* hostElements = new storeElement[N];
			for(int i=0; i<N; i++)
			{
				hostElements[i].metric = 1;
				hostElements[i].tag = i%20;
				hostElements[i].time = i;
				hostElements[i].value = 666.666;
			}
			storeQuery query;
			storeElement* deviceElements = nullptr;
			CUDA_CHECK_RETURN( cudaMalloc(&deviceElements, N*sizeof(storeElement)) );
			CUDA_CHECK_RETURN( cudaMemcpy(deviceElements, hostElements, N*sizeof(storeElement), cudaMemcpyHostToDevice) )

			// TEST
			size_t size = _queryCore->filterData(deviceElements, N*sizeof(storeElement), &query);

			// CHECK
			ASSERT_EQ(N*sizeof(storeElement), size);
			CUDA_CHECK_RETURN( cudaMemcpy(hostElements, deviceElements, size, cudaMemcpyDeviceToHost) )

			for(int i=0; i<N; i++)
			{
				EXPECT_EQ(1, hostElements[i].metric);
				EXPECT_EQ(i%20, hostElements[i].tag);
				EXPECT_EQ(i, hostElements[i].time);
				EXPECT_FLOAT_EQ(666.666, hostElements[i].value);
			}

			// CLEAN
			delete [] hostElements;
			CUDA_CHECK_RETURN( cudaFree(deviceElements) );
		}

		TEST_F(StoreQueryCoreTest, filterData_ExistingTags)
		{
			// PREPARE
			int N = 100;
			storeElement* hostElements = new storeElement[N];
			for(int i=0; i<N; i++)
			{
				hostElements[i].metric = 1;
				hostElements[i].tag = i%20;
				hostElements[i].time = 696969;
				hostElements[i].value = 666.666;
			}
			storeQuery query;
			query.tags.push_back(4);	// 5 times
			query.tags.push_back(12);	// 5 times
			query.tags.push_back(17);	// 5 times
			storeElement* deviceElements = nullptr;
			CUDA_CHECK_RETURN( cudaMalloc(&deviceElements, N*sizeof(storeElement)) );
			CUDA_CHECK_RETURN( cudaMemcpy(deviceElements, hostElements, N*sizeof(storeElement), cudaMemcpyHostToDevice) )

			// TEST
			size_t size = _queryCore->filterData(deviceElements, N*sizeof(storeElement), &query);

			// CHECK
			ASSERT_EQ(15*sizeof(storeElement), size);
			CUDA_CHECK_RETURN( cudaMemcpy(hostElements, deviceElements, size, cudaMemcpyDeviceToHost) )
			auto checkTagFunc = [&] (const int& tag)
				{
				if (tag == 4 || tag == 12 || tag == 17)
				    return ::testing::AssertionSuccess();
				  else
				    return ::testing::AssertionFailure() << "Expected: tag=4|12|17\nActual: tag=" << tag;
				};

			for(int i=0; i<15; i++)
			{
				EXPECT_EQ(1, hostElements[i].metric);
				EXPECT_TRUE(checkTagFunc(hostElements[i].tag));
				EXPECT_EQ(696969, hostElements[i].time);
				EXPECT_FLOAT_EQ(666.666, hostElements[i].value);
			}

			// CLEAN
			delete [] hostElements;
			CUDA_CHECK_RETURN( cudaFree(deviceElements) );
		}

		TEST_F(StoreQueryCoreTest, filterData_NonExistingTags)
		{
			// PREPARE
			int N = 100;
			storeElement* hostElements = new storeElement[N];
			for(int i=0; i<N; i++)
			{
				hostElements[i].metric = 1;
				hostElements[i].tag = i%20;
				hostElements[i].time = 696969;
				hostElements[i].value = 666.666;
			}
			storeQuery query;
			query.tags.push_back(24);	// 0 times
			query.tags.push_back(32);	// 0 times
			query.tags.push_back(7777);	// 0 times
			storeElement* deviceElements = nullptr;
			CUDA_CHECK_RETURN( cudaMalloc(&deviceElements, N*sizeof(storeElement)) );
			CUDA_CHECK_RETURN( cudaMemcpy(deviceElements, hostElements, N*sizeof(storeElement), cudaMemcpyHostToDevice) )

			// TEST
			size_t size = _queryCore->filterData(deviceElements, N*sizeof(storeElement), &query);

			// CHECK
			ASSERT_EQ(0, size);

			// CLEAN
			delete [] hostElements;
			CUDA_CHECK_RETURN( cudaFree(deviceElements) );
		}

		TEST_F(StoreQueryCoreTest, filterData_ExistingTags_FromTimePeriod)
		{
			// PREPARE
			int N = 100;
			storeElement* hostElements = new storeElement[N];
			for(int i=0; i<N; i++)
			{
				hostElements[i].metric = 1;
				hostElements[i].tag = i%20;
				hostElements[i].time = i;
				hostElements[i].value = 666.666;
			}

			storeQuery query;
			query.timePeriods.push_back(ullintPair{20,35});
			query.timePeriods.push_back(ullintPair{35,60});
			query.tags.push_back(4);	// 2 times in 20 to 60 time period
			query.tags.push_back(12);	// 2 times in 20 to 60 time period
			query.tags.push_back(17);	// 2 times in 20 to 60 time period
			storeElement* deviceElements = nullptr;
			CUDA_CHECK_RETURN( cudaMalloc(&deviceElements, N*sizeof(storeElement)) );
			CUDA_CHECK_RETURN( cudaMemcpy(deviceElements, hostElements, N*sizeof(storeElement), cudaMemcpyHostToDevice) )

			// EXPECTED RESULTS
			int expected_size = 6;

			// TEST
			size_t size = _queryCore->filterData(deviceElements, N*sizeof(storeElement), &query);

			// CHECK
			ASSERT_EQ(expected_size*sizeof(storeElement), size);
			CUDA_CHECK_RETURN( cudaMemcpy(hostElements, deviceElements, size, cudaMemcpyDeviceToHost) )
			auto checkTagFunc = [&] (const int& tag)
				{
				if (tag == 4 || tag == 12 || tag == 17)
					return ::testing::AssertionSuccess();
				  else
					return ::testing::AssertionFailure() << "Expected: tag=4|12|17\nActual: tag=" << tag;
				};

			for(int i=0; i<expected_size; i++)
			{
				EXPECT_EQ(1, hostElements[i].metric);
				EXPECT_TRUE(checkTagFunc(hostElements[i].tag));
				EXPECT_LE(20, hostElements[i].time);
				EXPECT_GE(60, hostElements[i].time);
				EXPECT_FLOAT_EQ(666.666, hostElements[i].value);
			}

			// CLEAN
			delete [] hostElements;
			CUDA_CHECK_RETURN( cudaFree(deviceElements) );
		}

	//selectData

		TEST_F(StoreQueryCoreTest, ExecuteQuery_SpecificTimeFrame_AllTags_NoAggregation)
		{
			// PREPARE
			createSimpleCharTestData();
			char* hostData;
			storeQuery query;
			query.aggregationType = AggregationType::None;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{64,127});

			// TEST
			size_t size = _queryCore->ExecuteQuery((void**)&hostData ,&query, dataLocationInfo);

			// CHECK
			ASSERT_EQ(64, size);
			for(unsigned long i=0; i<size; i++)
				EXPECT_EQ((char)((i+size)%256), hostData[i]);

			// CLEAN
			free( hostData );
		}

		TEST_F(StoreQueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_NoAggregation)
		{
			// PREPARE
			createTestDataWithStoreElements();
			storeElement* hostData;
			storeQuery query;
			query.aggregationType = AggregationType::None;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{0,STORE_QUERY_CORE_TEST_MEM_SIZE*sizeof(storeElement)});	// all
			query.tags.push_back(1);	// 52 elements
			query.tags.push_back(6);	// 51 elements
			query.tags.push_back(11);	// 51 elements
			query.tags.push_back(19);	// 51 elements
			query.timePeriods.push_back(ullintPair{1000,2000});	// all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{3000,4000}); // all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{9000,10240});// all: 124 elements, with good tags: 25

			// EXPECTED
			int expected_elements_count = 65;
			int expected_elements_size = expected_elements_count*sizeof(storeElement);
			auto checkTagFunc = [&] (const int& tag)
			{
				if (tag == 1 || tag == 6 || tag == 11 || tag == 19)
					return ::testing::AssertionSuccess();
				else
					return ::testing::AssertionFailure() << "Expected: tag=4|12|17\nActual: tag=" << tag;
			};
			auto checkTimeFunc = [&] (const ullint& time)
			{
				if ( (time > 1000 && time < 2000) ||
					 (time > 3000 && time < 4000) ||
					 (time > 9000 && time < 10240) )
					return ::testing::AssertionSuccess();
				else
					return ::testing::AssertionFailure() << "Expected: time in range (1000-2000) or (3000-4000) or (9000-10240)\nActual: tag=" << time;
			};


			// TEST
			size_t size = _queryCore->ExecuteQuery((void**)&hostData ,&query, dataLocationInfo);

			// CHECK
			ASSERT_EQ(expected_elements_size, size);
			for(int i=0; i < expected_elements_count; i++)
			{
				EXPECT_EQ(1, hostData[i].metric);
				EXPECT_TRUE(checkTagFunc(hostData[i].tag));
				EXPECT_TRUE(checkTimeFunc(hostData[i].time));
				EXPECT_FLOAT_EQ(3, hostData[i].value);
			}

			// CLEAN
			free( hostData );
		}

		TEST_F(StoreQueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation)
		{
			// PREPARE
			createTestDataWithStoreElements();
			storeElement* hostData;
			storeQuery query;
			query.aggregationType = AggregationType::Add;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{0,STORE_QUERY_CORE_TEST_MEM_SIZE*sizeof(storeElement)});	// all
			query.tags.push_back(1);	// 52 elements
			query.tags.push_back(6);	// 51 elements
			query.tags.push_back(11);	// 51 elements
			query.tags.push_back(19);	// 51 elements
			query.timePeriods.push_back(ullintPair{1000,2000});	// all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{3000,4000}); // all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{9000,10240});// all: 124 elements, with good tags: 25

			// EXPECTED
			int expected_elements_count = 1;
			int expected_elements_size = expected_elements_count*sizeof(storeElement);
			float expected_value = 65*3.0f;
			auto checkTagFunc = [&] (const int& tag)
			{
				if (tag == 1 || tag == 6 || tag == 11 || tag == 19)
					return ::testing::AssertionSuccess();
				else
					return ::testing::AssertionFailure() << "Expected: tag=4|12|17\nActual: tag=" << tag;
			};
			auto checkTimeFunc = [&] (const ullint& time)
			{
				if ( (time > 1000 && time < 2000) ||
					 (time > 3000 && time < 4000) ||
					 (time > 9000 && time < 10240) )
					return ::testing::AssertionSuccess();
				else
					return ::testing::AssertionFailure() << "Expected: time in range (1000-2000) or (3000-4000) or (9000-10240)\nActual: tag=" << time;
			};


			// TEST
			size_t size = _queryCore->ExecuteQuery((void**)&hostData ,&query, dataLocationInfo);

			// CHECK
			ASSERT_EQ(expected_elements_size, size);
			EXPECT_FLOAT_EQ(expected_value, hostData[0].value);

			// CLEAN
			free( hostData );
		}

	/***********************/
	/* AGGREGATION MATHODS */
	/***********************/

	//add

		TEST_F(StoreQueryCoreTest, add_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			storeElement* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->add(elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(StoreQueryCoreTest, add_EvenNumberOfValues)
		{
			// PREPARE
			int numberOfValues = 123;
			size_t dataSize = numberOfValues*sizeof(storeElement);
			storeElement* deviceData;
			cudaMalloc(&deviceData, dataSize);
			storeElement* hostData = new storeElement[numberOfValues];
			for(int i=0; i < numberOfValues; i++) hostData[i].value = 3;
			cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
			storeElement* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_sum = 3*123;

			// TEST
			size_t actual_size = _queryCore->add(deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_sum, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

		TEST_F(StoreQueryCoreTest, add_OddNumberOfValues)
		{
			// PREPARE
			int numberOfValues = 2000;
			size_t dataSize = numberOfValues*sizeof(storeElement);
			storeElement* deviceData;
			cudaMalloc(&deviceData, dataSize);
			storeElement* hostData = new storeElement[numberOfValues];
			for(int i=0; i < numberOfValues; i++) hostData[i].value = 4.2f;
			cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
			storeElement* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_sum = 4.2f*2000;

			// TEST
			size_t actual_size = _queryCore->add(deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_sum, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

	//max

		TEST_F(StoreQueryCoreTest, max_Empty)
		{

		}

		TEST_F(StoreQueryCoreTest, max_Positive)
		{

		}

		TEST_F(StoreQueryCoreTest, max_Negative)
		{

		}

	//min

		TEST_F(StoreQueryCoreTest, min_Empty)
		{

		}

		TEST_F(StoreQueryCoreTest, min_Positive)
		{

		}

		TEST_F(StoreQueryCoreTest, min_Negative)
		{

		}


} /* namespace store */
} /* namespace ddj */
