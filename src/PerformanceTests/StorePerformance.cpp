#include "StorePerformance.h"

namespace ddj {
namespace task {

	using namespace std::chrono;
	using namespace ddj::store;

	INSTANTIATE_TEST_CASE_P(StorePerformanceInst,
						StorePerformance,
						::testing::Values(1000000));

	TEST_P(StorePerformance, InsertData_EqualElements)
	{
		int N = GetParam();
		int X = 5;
		int threadCount = _config->GetIntValue("THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		for(int i=0; i<X; i++)
		{
			storeElement* elem = new storeElement(1, 1, 1, 69.69);
			Task_Pointer tp(new Task(1, Insert, elem, N, &_taskCond));
			auto start = system_clock::now();
			for(int i=0; i<N; i++)
				_storeController->ExecuteTask(tp);
			boost::mutex mutex;
			boost::mutex::scoped_lock lock(mutex);
			while(1)
			{
				_taskCond.wait(lock);
				if(tp->IsCompleated())
					break;
			}
			auto end = system_clock::now();
			D += end - start;
		}
		printf("[InsertEqualElem|%d] (%f ms)\n", N, D.count()/X);
		_resultFile << "InsertEqualElem " << N << " " << threadCount << " "
				<< bufferSize << " " << D.count()/X << std::endl;
	}

} /* namespace task */
} /* namespace ddj */
