/*
 * IngegralPerformance.h
 *
 *  Created on: 05-02-2014
 *      Author: ghash
 */

#ifndef INGEGRALPERFORMANCE_H_
#define INGEGRALPERFORMANCE_H_

#include "../Query/QueryAggregation.h"
#include "../Query/QueryAggregationResults.cuh"
#include "../Cuda/CudaController.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaCommons.h"
#include "../Cuda/CudaIncludes.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <fstream>

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../Core/helper_cuda.h"

namespace ddj {
namespace query {

	class IntegralPerformance : public ::testing::TestWithParam<int>
	{
		protected:
			IntegralPerformance()
			{
				_config = Config::GetInstance();
				CudaCommons cudaC;
				cudaC.SetCudaDeviceWithMaxFreeMem();
				_resultFile.open("./src/PerformanceTests/Results/IntegralPerformanceResult.txt",
						std::ofstream::app);
				if(!_resultFile.is_open())
				{
					LOG4CPLUS_FATAL(_logger, LOG4CPLUS_TEXT("Unable to open file"));
				}
				_queryAggregation = new QueryAggregation();
			}

			void LogDeviceMemory()
			{
				CudaCommons cudaC;
				size_t free;
				size_t total;
				cudaC.GetMemoryCount(&free, &total);
				LOG4CPLUS_INFO_FMT(_logger, LOG4CPLUS_TEXT("Device memory: %f/%f MB"), (float)free/1048576, (float)total/1048576);
			}

			virtual ~IntegralPerformance()
			{
				this->_resultFile.close();
				delete _queryAggregation;
			}

			Config* _config;
			ofstream _resultFile;
			Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("test"));
			QueryAggregation* _queryAggregation;
	};

} /* namespace query */
} /* namespace ddj */
#endif /* INGEGRALPERFORMANCE_H_ */
