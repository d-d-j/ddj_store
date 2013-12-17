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

	TEST_F(StoreQueryCoreTest, ThrustVersion)
	{
		int major = THRUST_MAJOR_VERSION;
		int minor = THRUST_MINOR_VERSION;

		LOG4CPLUS_INFO(this->_logger, "Thrust version: " << major << "." << minor);

		EXPECT_EQ(1, major);
		EXPECT_EQ(7, minor);
	}

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

	TEST_F(StoreQueryCoreTest, mapData_AllData)
	{
		// PREPARE
		char* data;

		// TEST
		size_t size = _queryCore->mapData((void**)&data);

		// CHECK
		ASSERT_EQ(STORE_QUERY_CORE_TEST_MEM_SIZE, size);
		for(unsigned long i=0; i<size; i++)
			if(data[i]!=(char)(i%256))
				ADD_FAILURE();

	}

} /* namespace store */
} /* namespace ddj */