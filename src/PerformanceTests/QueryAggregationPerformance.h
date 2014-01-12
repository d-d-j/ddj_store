/*
 * QueryAggregationPerformance.h
 *
 *  Created on: 09-01-2014
 *      Author: ghash
 */

#ifndef QUERYAGGREGATIONPERFORMANCE_H_
#define QUERYAGGREGATIONPERFORMANCE_H_

#include "../Query/QueryAggregation.h"
#include "../Query/AggregationResults.cuh"
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

	class QueryAggregationPerformance : public ::testing::TestWithParam<int>
	{
	protected:
		QueryAggregationPerformance()
		{
			SetCudaDeviceWithMaxFreeMem();
			LogDeviceMemory();
			_resultFile.open("./src/PerformanceTests/Results/QueryAggregationPerformanceResult.txt",
					std::ofstream::app);
			if(!_resultFile.is_open())
			{
				LOG4CPLUS_FATAL(_logger, LOG4CPLUS_TEXT("Unable to open file"));
			}
			_queryAggregation = new QueryAggregation();
		}

		void SetCudaDeviceWithMaxFreeMem()
		{
			const char* argv = "";
			int deviceId = 0;
			CudaCommons cudaC;
			int devCount = cudaC.CudaGetDevicesCount();
			size_t free;
			size_t total;
			size_t max_free = 0;
			//cudaC.GetMemoryCount(&free, &total);
			for(int i=0; i<devCount; i++)
			{
				cudaSetDevice(i);
				cudaC.GetMemoryCount(&free, &total);
				if(free > max_free)
				{
					max_free = free;
					deviceId = i;
				}
			}
			cudaSetDevice(deviceId);
			LOG4CPLUS_INFO(_logger, LOG4CPLUS_TEXT("Setting cuda device ") << deviceId);
		}

		void LogDeviceMemory()
		{
			CudaCommons cudaC;
			size_t free;
			size_t total;
			cudaC.GetMemoryCount(&free, &total);
			LOG4CPLUS_INFO_FMT(_logger, LOG4CPLUS_TEXT("Device memory: %f/%f MB"), (float)free/1048576, (float)total/1048576);
		}

		virtual void TearDown()
		{
			LogDeviceMemory();
		}

		virtual ~QueryAggregationPerformance()
		{
			this->_resultFile.close();
			delete _queryAggregation;
		}

		ofstream _resultFile;
		Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("test"));
		QueryAggregation* _queryAggregation;
	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYAGGREGATIONPERFORMANCE_H_ */
