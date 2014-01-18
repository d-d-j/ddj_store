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

		// CLEAN
		delete [] testArray;
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

		// CLEAN
		delete [] testArray;
	}

	void QueryCoreTest::createTestDataWithStoreElements_100Elem()
	{
		void* mainMemoryPointer = _cudaController->GetMainMemoryPointer();
		int size = 100;
		storeElement* testArray = new storeElement[size];
		for(int i=0; i<size; i++)
		{
			testArray[i].metric = 1;
			testArray[i].tag = i%20;
			testArray[i].time = i;
			testArray[i].value = 666.666;
		}

		CUDA_CHECK_RETURN( cudaMemcpy(mainMemoryPointer, testArray, size*sizeof(storeElement), cudaMemcpyHostToDevice) );
		_cudaController->SetMainMemoryOffset(size*sizeof(storeElement));

		// CLEAN
		delete [] testArray;
	}

	size_t QueryCoreTest::createCompressedTestDataWithStoreElementsOneTrunk()
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
		storeElement* device_elementsToCompress;
		CUDA_CHECK_RETURN( cudaMalloc(&device_elementsToCompress, size*sizeof(storeElement)) );
		CUDA_CHECK_RETURN( cudaMemcpy(device_elementsToCompress, testArray, size*sizeof(storeElement), cudaMemcpyHostToDevice) );

		//Compress data and place it in main memory
		void* device_compressedElements;
		compression::Compression c;
		size_t compressedDataSize = c.CompressTrunk(device_elementsToCompress, size*sizeof(storeElement), &device_compressedElements);
		CUDA_CHECK_RETURN( cudaMemcpy(mainMemoryPointer, device_compressedElements, compressedDataSize, cudaMemcpyDeviceToDevice) );

		_cudaController->SetMainMemoryOffset(compressedDataSize);

		// CLEAN
		delete [] testArray;
		CUDA_CHECK_RETURN( cudaFree(device_elementsToCompress) );
		CUDA_CHECK_RETURN( cudaFree(device_compressedElements) );

		return compressedDataSize;
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
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{64,127});	// length 64
			dataLocationInfo.push_back(ullintPair{256,383});	// length 128
			dataLocationInfo.push_back(ullintPair{601,728});	// length 128

			// TEST
			size_t size = _queryCore->mapData((void**)&deviceData, &dataLocationInfo);

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

		TEST_F(QueryCoreTest, filterData_inTrunks_Empty_Trunk)
		{
			// PREPARE
			int N = 100;
			storeElement* hostElements = new storeElement[N];
			for(int i=0; i<N; i++)
			{
				hostElements[i].metric = 1;
				if(i>=40 && i<60) hostElements[i].tag = 20;
				else hostElements[i].tag = i%20;
				hostElements[i].time = 696969;
				hostElements[i].value = 666.666;
			}
			// COPY ELEMENTS TO DEVICE
			storeElement* deviceElements = nullptr;
			CUDA_CHECK_RETURN( cudaMalloc(&deviceElements, N*sizeof(storeElement)) );
			CUDA_CHECK_RETURN( cudaMemcpy(deviceElements, hostElements, N*sizeof(storeElement), cudaMemcpyHostToDevice) )

			// CREATE QUERY
			Query query;
			// AGGREGATION
			query.aggregationType = AggregationType::Integral;
			// TAGS
			query.tags.push_back(4);	// 4 elements
			query.tags.push_back(12);	// 4 elements
			query.tags.push_back(17);	// 4 elements

			// CREATE DATA LOCATION INFO
			size_t elemSize = sizeof(storeElement);
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{0*elemSize,40*elemSize-1});	// 6 elements
			dataLocationInfo.push_back(ullintPair{40*elemSize,60*elemSize-1});	// 0 elements
			dataLocationInfo.push_back(ullintPair{60*elemSize,100*elemSize-1});// 6 elements

			// EXPECTED RESULT
			int expected_elements_count = 12;
			size_t expected_size = expected_elements_count*sizeof(storeElement);

			// TEST
			size_t size = _queryCore->filterData(deviceElements, N*sizeof(storeElement), &query, &dataLocationInfo);

			// CHECK
			ASSERT_EQ(expected_size, size);
			CUDA_CHECK_RETURN( cudaMemcpy(hostElements, deviceElements, size, cudaMemcpyDeviceToHost) )
			auto checkTagFunc = [&] (const int& tag)
				{
				if (tag == 4 || tag == 12 || tag == 17)
					return ::testing::AssertionSuccess();
				  else
					return ::testing::AssertionFailure() << "Expected: tag=4|12|17\nActual: tag=" << tag;
				};

			for(int i=0; i<expected_elements_count; i++)
			{
				EXPECT_EQ(1, hostElements[i].metric);
				EXPECT_TRUE(checkTagFunc(hostElements[i].tag));
				EXPECT_EQ(696969, hostElements[i].time);
				EXPECT_FLOAT_EQ(666.666, hostElements[i].value);
			}

			// CHECK DATA LOCATION INFO
			ASSERT_EQ(2, dataLocationInfo.size());
			EXPECT_EQ(0*elemSize, dataLocationInfo[0].first);
			EXPECT_EQ(6*elemSize-1, dataLocationInfo[0].second);
			EXPECT_EQ(6*elemSize, dataLocationInfo[1].first);
			EXPECT_EQ(12*elemSize-1, dataLocationInfo[1].second);

			// CLEAN
			delete [] hostElements;
			CUDA_CHECK_RETURN( cudaFree(deviceElements) );
		}

	//mapAndFilterData

		TEST_F(QueryCoreTest, mapData_and_filterData_InTrunks_WithExistingTags_FromTimePeriod)
		{
			///////////////////////
			//////// PREPARE //////
			///////////////////////
			createTestDataWithStoreElements_100Elem();
			// CREATE QUERY
			Query query;
			// AGGREGATION
			query.aggregationType = AggregationType::Integral;
			// TIME PERIODS
			query.timePeriods.push_back(ullintPair{20,35});		// 15 elements
			query.timePeriods.push_back(ullintPair{35,60});		// 25 elements
			// TAGS
			query.tags.push_back(4);	// 2 times in 20 to 60 time period
			query.tags.push_back(12);	// 2 times in 20 to 60 time period
			query.tags.push_back(17);	// 2 times in 20 to 60 time period

			// CREATE DATA LOCATION INFO
			size_t elemSize = sizeof(storeElement);
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{15*elemSize,30*elemSize-1});	// 15 elements in trunk
			dataLocationInfo.push_back(ullintPair{30*elemSize,45*elemSize-1}); // 15 elements in trunk
			dataLocationInfo.push_back(ullintPair{45*elemSize,60*elemSize-1}); // 15 elements in trunk

			// EXPECTED RESULTS
			int expected_size = 6;
			storeElement* hostElements = nullptr;
			storeElement* deviceElements = nullptr;

			// TEST MAP
			size_t mappedDataSize = _queryCore->mapData((void**)&deviceElements, &dataLocationInfo);

			// CHECK MAP
			ASSERT_EQ(45*elemSize, mappedDataSize);
			EXPECT_EQ(0*elemSize, dataLocationInfo[0].first);
			EXPECT_EQ(15*elemSize-1, dataLocationInfo[0].second);
			EXPECT_EQ(15*elemSize, dataLocationInfo[1].first);
			EXPECT_EQ(30*elemSize-1, dataLocationInfo[1].second);
			EXPECT_EQ(30*elemSize, dataLocationInfo[2].first);
			EXPECT_EQ(45*elemSize-1, dataLocationInfo[2].second);

			// TEST FILTER
			size_t size = _queryCore->filterData(deviceElements, mappedDataSize, &query, &dataLocationInfo);

			// CHECK FILTER
			ASSERT_EQ(expected_size*sizeof(storeElement), size);
			hostElements = new storeElement[expected_size];
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
			// CHECK DATA LOCATION INFO AFTER FILTER
			EXPECT_EQ(0*elemSize, dataLocationInfo[0].first);
			EXPECT_EQ(1*elemSize-1, dataLocationInfo[0].second);
			EXPECT_EQ(1*elemSize, dataLocationInfo[1].first);
			EXPECT_EQ(4*elemSize-1, dataLocationInfo[1].second);
			EXPECT_EQ(4*elemSize, dataLocationInfo[2].first);
			EXPECT_EQ(6*elemSize-1, dataLocationInfo[2].second);

			// CLEAN
			delete [] hostElements;
			CUDA_CHECK_RETURN( cudaFree(deviceElements) );
		}

	//selectData without compression

		TEST_F(QueryCoreTest, ExecuteQuery_SpecificTimeFrame_AllTags_NoAggregation)
		{
			// PREPARE
			createSimpleCharTestData();
			char* hostData;
			Query query;
			query.aggregationType = AggregationType::None;
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{64,127});

			// TEST
			_queryCore->_enableCompression = false;
			size_t size = _queryCore->ExecuteQuery((void**)&hostData ,&query, &dataLocationInfo);

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

			Query query;
			query.aggregationType = AggregationType::None;
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{0,QUERY_CORE_TEST_MEM_SIZE*sizeof(storeElement)});	// all
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
			storeElement* result;


			// TEST
			_queryCore->_enableCompression = false;
			size_t size = _queryCore->ExecuteQuery((void**)&result ,&query, &dataLocationInfo);

			// CHECK
			ASSERT_EQ(expected_elements_size, size);
			for(int i=0; i < expected_elements_count; i++)
			{
				EXPECT_EQ(1, result[i].metric);
				EXPECT_TRUE(checkTagFunc(result[i].tag));
				EXPECT_TRUE(checkTimeFunc(result[i].time));
				EXPECT_FLOAT_EQ(3, result[i].value);
			}

			// CLEAN
			free( result );
		}

		TEST_F(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation)
		{
			// PREPARE
			createTestDataWithStoreElements();

			Query query;
			query.aggregationType = AggregationType::Sum;
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{0,QUERY_CORE_TEST_MEM_SIZE*sizeof(storeElement)});	// all
			query.tags.push_back(1);	// 52 elements
			query.tags.push_back(6);	// 51 elements
			query.tags.push_back(11);	// 51 elements
			query.tags.push_back(19);	// 51 elements
			query.timePeriods.push_back(ullintPair{1000,2000});	// all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{3000,4000}); // all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{9000,10240});// all: 124 elements, with good tags: 25

			// EXPECTED
			int expected_elements_count = 1;
			int expected_elements_size = expected_elements_count*sizeof(results::sumResult);
			float expected_sum = 65*3.0f;
			results::sumResult* result;

			// TEST
			_queryCore->_enableCompression = false;
			size_t size = _queryCore->ExecuteQuery((void**)&result ,&query, &dataLocationInfo);

			// CHECK
			ASSERT_EQ(expected_elements_size, size);
			EXPECT_FLOAT_EQ(expected_sum, result->sum);

			// CLEAN
			free( result );
		}

	//selectData with compression

		TEST_F(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation_OneTrunk_Compression)
		{
			// PREPARE
			size_t sizeOfData = createCompressedTestDataWithStoreElementsOneTrunk();

			Query query;
			query.aggregationType = AggregationType::Sum;
			boost::container::vector<ullintPair> dataLocationInfo;
			dataLocationInfo.push_back(ullintPair{0,sizeOfData-1});	// all

			query.tags.push_back(1);	// 52 elements
			query.tags.push_back(6);	// 51 elements
			query.tags.push_back(11);	// 51 elements
			query.tags.push_back(19);	// 51 elements
			query.timePeriods.push_back(ullintPair{1000,2000});	// all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{3000,4000}); // all: 100 elements, with good tags: 20
			query.timePeriods.push_back(ullintPair{9000,10240});// all: 124 elements, with good tags: 25

			// EXPECTED
			int expected_elements_count = 1;
			int expected_elements_size = expected_elements_count*sizeof(results::sumResult);
			float expected_sum = 65*3.0f;
			results::sumResult* result;

			// TEST
			_queryCore->_enableCompression = true;
			size_t size = _queryCore->ExecuteQuery((void**)&result ,&query, &dataLocationInfo);

			// CHECK
			ASSERT_EQ(expected_elements_size, size);
			EXPECT_FLOAT_EQ(expected_sum, result->sum);

			// CLEAN
			free( result );
		}


} /* namespace query */
} /* namespace ddj */
