#include "QueryCoreTest.h"

namespace ddj {
namespace query {

	void QueryCoreTest::createSimpleCharTestData()
	{
		void* mainMemoryPointer = _cudaController->GetMainMemoryPointer();
		size_t size = QUERY_CORE_TEST_MEM_SIZE;
		char* testArray = new char[size];
		// FILL TAST ARRAY WITH CHARACTERS
		for(unsigned long i=0; i<size; i++)
			testArray[i] = (char)(i%256);

		CUDA_CHECK_RETURN( cudaMemcpy(mainMemoryPointer, testArray, size, cudaMemcpyHostToDevice) );
		_cudaController->SetMainMemoryOffset(size);
	}

	void QueryCoreTest::createTestDataWithStoreElements()
	{
		void* mainMemoryPointer = _cudaController->GetMainMemoryPointer();
		size_t size = QUERY_CORE_TEST_MEM_SIZE;
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
	TEST_F(QueryCoreTest, ThrustVersion)
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

		TEST_F(QueryCoreTest, mapData_AllData)
		{
			// PREPARE
			createSimpleCharTestData();
			char* deviceData;
			char* hostData;

			// TEST
			size_t size = _queryCore->mapData((void**)&deviceData);

			// CHECK
			ASSERT_EQ(QUERY_CORE_TEST_MEM_SIZE, size);

			hostData = new char[size];
			CUDA_CHECK_RETURN( cudaMemcpy(hostData, deviceData, size, cudaMemcpyDeviceToHost) );

			for(unsigned long i=0; i<size; i++)
				EXPECT_EQ((char)(i%256), hostData[i]);

			// CLEAN
			delete [] hostData;
			CUDA_CHECK_RETURN( cudaFree(deviceData) );
		}

		TEST_F(QueryCoreTest, mapData_ChooseOneTrunk)
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

		TEST_F(QueryCoreTest, mapData_ChooseManyTrunks)
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

		TEST_F(QueryCoreTest, filterData_AllData)
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
			Query query;
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

		TEST_F(QueryCoreTest, filterData_ExistingTags)
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
			Query query;
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

		TEST_F(QueryCoreTest, filterData_NonExistingTags)
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
			Query query;
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

		TEST_F(QueryCoreTest, filterData_ExistingTags_FromTimePeriod)
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

			Query query;
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

		TEST_F(QueryCoreTest, ExecuteQuery_SpecificTimeFrame_AllTags_NoAggregation)
		{
			// PREPARE
			createSimpleCharTestData();
			char* hostData;
			Query query;
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

		TEST_F(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_NoAggregation)
		{
			// PREPARE
			createTestDataWithStoreElements();
			storeElement* hostData;
			Query query;
			query.aggregationType = AggregationType::None;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{0,QUERY_CORE_TEST_MEM_SIZE*sizeof(storeElement)});	// all
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

		TEST_F(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation)
		{
			// PREPARE
			createTestDataWithStoreElements();
			storeElement* hostData;
			Query query;
			query.aggregationType = AggregationType::Add;
			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();
			dataLocationInfo->push_back(ullintPair{0,QUERY_CORE_TEST_MEM_SIZE*sizeof(storeElement)});	// all
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

		TEST_F(QueryCoreTest, add_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			void* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->add(elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(QueryCoreTest, add_EvenNumberOfValues)
		{
			// PREPARE
			int numberOfValues = 123;
			size_t dataSize = numberOfValues*sizeof(storeElement);
			storeElement* deviceData;
			cudaMalloc(&deviceData, dataSize);
			storeElement* hostData = new storeElement[numberOfValues];
			for(int i=0; i < numberOfValues; i++) hostData[i].value = 3;
			cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
			void* deviceResult;
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

		TEST_F(QueryCoreTest, add_OddNumberOfValues)
		{
			// PREPARE
			int numberOfValues = 2000;
			size_t dataSize = numberOfValues*sizeof(storeElement);
			storeElement* deviceData;
			cudaMalloc(&deviceData, dataSize);
			storeElement* hostData = new storeElement[numberOfValues];
			for(int i=0; i < numberOfValues; i++) hostData[i].value = 4.2f;
			cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
			void* deviceResult;
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

		TEST_F(QueryCoreTest, max_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			void* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->max(elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(QueryCoreTest, max_Positive)
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
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_max = 8100.0f;

			// TEST
			size_t actual_size = _queryCore->max(deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_max, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

		TEST_F(QueryCoreTest, max_Negative)
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
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_max = 3636181.0f;

			// TEST
			size_t actual_size = _queryCore->max(deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_max, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

	//min

		TEST_F(QueryCoreTest, min_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			void* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->min(elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(QueryCoreTest, min_Positive)
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
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_min = 3.0f;

			// TEST
			size_t actual_size = _queryCore->min(deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_min, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

		TEST_F(QueryCoreTest, min_Negative)
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
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_min = -8100.0f;

			// TEST
			size_t actual_size = _queryCore->min(deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_min, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

	//average

		TEST_F(QueryCoreTest, average_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			void* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::Average](elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(QueryCoreTest, average_Linear)
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
			results::averageResult* hostResult;

			// EXPECTED
			size_t expected_size = sizeof(results::averageResult);
			float expected_sum = 0.0f;
			int32_t expected_count = 2001;

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::Average](deviceData, dataSize, (void**)&hostResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_FLOAT_EQ(expected_sum, hostResult->sum);
			EXPECT_EQ(expected_count, hostResult->count);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

		TEST_F(QueryCoreTest, average_Sinusoidal)
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
			results::averageResult* hostResult;

			// EXPECTED
			size_t expected_size = sizeof(results::averageResult);
			float expected_sum = 0.0f;
			int expected_count = 2001;

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::Average](deviceData, dataSize, (void**)&hostResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_FLOAT_EQ(expected_sum, hostResult->sum);
			EXPECT_EQ(expected_count, hostResult->count);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

	//stdDeviation

		TEST_F(QueryCoreTest, stdDeviation_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			void* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::StdDeviation](elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(QueryCoreTest, stdDeviation_Simple)
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
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			/*
			 * number of values: n = 4
			 * values: {5, 6, 8, 9}
			 * average: a = 7
			 * standard deviation: s = sqrt[1/(n-1) * SUM[i=0 to 3: (values[i] - a)^2] ]
			 * s = sqrt[1/3 * (4+1+1+4)] = sqrt[10/3]
			 */
			size_t expected_size = sizeof(storeElement);
			float expected_stdDeviation = std::sqrt(10.0f/3.0f);

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::StdDeviation](deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_stdDeviation, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

		TEST_F(QueryCoreTest, stdDeviation_Linear)
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
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			/*
			 * number of values: n = 2001
			 * values: {0, 1,..., 1999, 2000}
			 * average: a = 1000
			 * standard deviation: s = sqrt[1/2000 * SUM[i=0 to 2000: (values[i] - 1000)^2] ]
			 * s = sqrt[667667/2]
			 */
			size_t expected_size = sizeof(storeElement);
			float expected_stdDeviation = std::sqrt(667667.0f/2.0f);

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::StdDeviation](deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_stdDeviation, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

	//count

		TEST_F(QueryCoreTest, count_Empty)
		{
			// PREPARE
			storeElement* elements = nullptr;
			size_t dataSize = 0;
			void* result;

			// EXPECTED
			size_t expected_size = 0;

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::Count](elements, dataSize, &result);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			EXPECT_EQ(nullptr, result);
		}

		TEST_F(QueryCoreTest, count_NonEmpty)
		{
			// PREPARE
			int numberOfValues = 2001;
			size_t dataSize = numberOfValues*sizeof(storeElement);
			storeElement* deviceData;
			cudaMalloc(&deviceData, dataSize);
			storeElement* hostData = new storeElement[numberOfValues];
			for(int i=0; i < numberOfValues; i++) {
				hostData[i].value = 2*i;
			}
			cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
			void* deviceResult;
			storeElement hostResult;

			// EXPECTED
			size_t expected_size = sizeof(storeElement);
			float expected_count = 2001.0f;

			// TEST
			size_t actual_size = _queryCore->_aggregationFunctions[AggregationType::Count](deviceData, dataSize, &deviceResult);

			// CHECK
			ASSERT_EQ(expected_size, actual_size);
			cudaMemcpy(&hostResult, deviceResult, sizeof(storeElement), cudaMemcpyDeviceToHost);
			EXPECT_FLOAT_EQ(expected_count, hostResult.value);

			// CLEAN
			delete [] hostData;
			cudaFree(deviceData);
		}

	//variance

		TEST_F(QueryCoreTest, variance_Empty)
		{

		}

		TEST_F(QueryCoreTest, variance_Simple)
		{

		}

		TEST_F(QueryCoreTest, variance_Linear)
		{

		}

} /* namespace query */
} /* namespace ddj */
