#ifndef STOREQUERYCORETEST_H_
#define QUERYCORETEST_H_

#define _USE_MATH_DEFINES
#define QUERY_CORE_TEST_MEM_SIZE 1024

#include "../Query/QueryCore.h"
#include "../Query/AggregationResults.h"
#include "../Cuda/CudaController.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaIncludes.h"
#include <gtest/gtest.h>
#include <cmath>

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../Core/helper_cuda.h"
#include <thrust/version.h>

namespace ddj {
namespace query {

using namespace store;

	class QueryCoreTest : public ::testing::Test
	{
		int _devId;

	protected:
		QueryCoreTest()
		{
			const char* argv = "";
			_devId = findCudaDevice(0, &argv);
			_cudaController = nullptr;
			_queryCore = nullptr;
		}

		virtual void SetUp()
		{
			_cudaController = new CudaController(3,3,_devId);
			_queryCore = new QueryCore(_cudaController);
		}
		virtual void TearDown()
		{
			delete _cudaController;
			delete _queryCore;
		}

		void createSimpleCharTestData();
		void createTestDataWithStoreElements();

		QueryCore* _queryCore;
		CudaController* _cudaController;
	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYCORETEST_H_ */
