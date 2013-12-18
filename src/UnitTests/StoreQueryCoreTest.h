/*
 * StoreQueryCoreTest.h
 *
 *  Created on: 17-12-2013
 *      Author: ghash
 */

#ifndef STOREQUERYCORETEST_H_
#define STOREQUERYCORETEST_H_

#include "../Store/StoreQueryCore.h"
#include "../Cuda/CudaController.h"
#include "../Core/Logger.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace ddj {
namespace store {

	class StoreQueryCoreTest : public ::testing::Test
	{
	protected:
		StoreQueryCoreTest()
		{
			_cudaController = new CudaController(3,3,0);
			this->createTestData();
			_queryCore = nullptr;
		}
		virtual ~StoreQueryCoreTest()
		{
			delete _cudaController;
		}

		virtual void SetUp()
		{
			_queryCore = new StoreQueryCore(_cudaController);
		}
		virtual void TearDown()
		{
			delete _queryCore;
		}

		void createTestData();

		StoreQueryCore* _queryCore;
		CudaController* _cudaController;
	};

} /* namespace store */
} /* namespace ddj */
#endif /* STOREQUERYCORETEST_H_ */
