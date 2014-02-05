#ifndef INSERTPERFORMANCE_H_
#define INSERTPERFORMANCE_H_

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
#include <random>
#include <limits>

namespace ddj {
namespace task {

	class InsertPerformance : public ::testing::TestWithParam<int>
	{
	protected:
		InsertPerformance()
		{
			_config = Config::GetInstance();
			CudaCommons cudaC;
			this->_devId = cudaC.SetCudaDeviceWithMaxFreeMem();
			_storeController = new store::StoreController(this->_devId);
			_taskMonitor = new task::TaskMonitor(&_taskCond);
			_resultFile.open("./src/PerformanceTests/Results/InsertPerformanceResult.txt", std::ofstream::app);
			if(!_resultFile.is_open())
			{
				LOG4CPLUS_FATAL(_logger, LOG4CPLUS_TEXT("Unable to open file"));
			}
		}

		virtual ~InsertPerformance()
		{
			this->_resultFile.close();
			delete _storeController;
			delete _taskMonitor;
		}


		int _devId;
		ofstream _resultFile;
		Logger _logger = Logger::getInstance(LOG4CPLUS_TEXT("test"));
		Config* _config;
		store::StoreController* _storeController;
		TaskMonitor* _taskMonitor;
		boost::condition_variable _taskCond;
	};

} /* namespace task */
} /* namespace ddj */
#endif /* INSERTPERFORMANCE_H_ */
