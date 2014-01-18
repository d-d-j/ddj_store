/*
 * StorePerformance.h
 *
 *  Created on: 11-01-2014
 *      Author: ghash
 */

#ifndef STOREPERFORMANCE_H_
#define STOREPERFORMANCE_H_

#include "../Store/StoreController.h"
#include "../Task/TaskMonitor.h"
#include "../Core/Logger.h"
#include "../Core/Config.h"
#include "../Cuda/CudaCommons.h"
#include "../Cuda/CudaIncludes.h"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

namespace ddj {
namespace task {

	class StorePerformance : public ::testing::TestWithParam<int>
	{
	protected:
		StorePerformance()
		{
			_config = Config::GetInstance();
			CudaCommons cudaC;
			int devId = cudaC.SetCudaDeviceWithMaxFreeMem();
			_storeController = new store::StoreController(devId);
			_taskMonitor = new task::TaskMonitor(&_taskCond);
			_resultFile.open("./src/PerformanceTests/Results/StorePerformanceResult.txt", std::ofstream::app);
			if(!_resultFile.is_open())
			{
				LOG4CPLUS_FATAL(_logger, LOG4CPLUS_TEXT("Unable to open file"));
			}
		}

		virtual ~StorePerformance()
		{
			this->_resultFile.close();
			delete _storeController;
			delete _taskMonitor;
		}


		ofstream _resultFile;
		Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("test"));
		Config* _config;
		store::StoreController* _storeController;
		TaskMonitor* _taskMonitor;
		boost::condition_variable _taskCond;
	};

} /* namespace task */
} /* namespace ddj */
#endif /* STOREPERFORMANCE_H_ */
