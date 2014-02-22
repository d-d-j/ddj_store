#include "InsertPerformance.h"

namespace ddj {
namespace task {

	using namespace std::chrono;
	using namespace ddj::store;

	INSTANTIATE_TEST_CASE_P(InsertPerformanceInst, InsertPerformance, ::testing::Values(1000000));

	TEST_P(InsertPerformance, InsertData_EqualElements)
	{
		int N = GetParam();
		int X = 5;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		for(int i=0; i<X; i++)
		{
			storeElement* elem = new storeElement(1, 1, i, 69.69);
			Task_Pointer tp(new Task(1, Insert, elem, N, &_taskCond));
			auto start = system_clock::now();
			for(int j=0; j<N; j++)
				_storeController->ExecuteTask(tp);
			boost::mutex mutex;
			boost::mutex::scoped_lock lock(mutex);
			while(1)
			{
				_taskCond.wait(lock);
				if(tp->IsCompleted())
					break;
			}
			auto end = system_clock::now();
			D += end - start;

			delete this->_storeController;
			_storeController = new store::StoreController(this->_devId);
		}
		printf("[InsertEqualElem|%d] (%f ms)\n", N, D.count()/X);
		_resultFile << "InsertEqualElem " << N << " " << threadCount << " "
				<< bufferSize << " " << D.count()/X
				<< " " << N/(D.count()/X/1000.0f) << std::endl;
	}

//	TEST_P(InsertPerformance, InserData_LinearElements)
//	{
//		int N = GetParam();
//		int X = 5;
//		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
//		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
//		duration<double, milli> D;
//		for(int i=0; i<X; i++)
//		{
//			auto start = system_clock::now();
//			for(int j=0; j<N; j++)
//			{
//				storeElement* elem = new storeElement(i%3000, i%15, i, i*69.69);
//				Task_Pointer tp(new Task(1, Insert, elem, N, &_taskCond));
//				_storeController->ExecuteTask(tp);
//			}
//			delete this->_storeController;
//			auto end = system_clock::now();
//			D += end - start;
//			_storeController = new store::StoreController(this->_devId);
//		}
//		printf("[InserDataLinearElements|%d] (%f ms)\n", N, D.count()/X);
//		_resultFile << "InsertLinearElem " << N << " " << threadCount << " "
//				<< bufferSize << " " << D.count()/X
//				<< " " << N/(D.count()/X/1000.0f) << std::endl;
//	}

} /* namespace task */
} /* namespace ddj */
