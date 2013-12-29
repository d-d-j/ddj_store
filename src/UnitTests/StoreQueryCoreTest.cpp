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

	void StoreQueryCoreTest::createTestData()
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

		}

	/***********************/
	/* AGGREGATION MATHODS */
	/***********************/

	//add

		TEST_F(StoreQueryCoreTest, add_Empty)
		{

		}

		TEST_F(StoreQueryCoreTest, add_EvenNumberOfValues)
		{

		}

		TEST_F(StoreQueryCoreTest, add_OddNumberOfValues)
		{

		}

	//average

		TEST_F(StoreQueryCoreTest, average_Empty)
		{

		}

		TEST_F(StoreQueryCoreTest, average_Positive)
		{

		}

		TEST_F(StoreQueryCoreTest, average_Negative)
		{

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
