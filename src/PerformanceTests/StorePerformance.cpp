#include "StorePerformance.h"

namespace ddj {
namespace task {

	using namespace std::chrono;
	using namespace ddj::store;

	INSTANTIATE_TEST_CASE_P(StorePerformanceInst,
						StorePerformance,
						::testing::Values(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000));

	void insertDataSin(StoreController* controller, int metric, int tag, boost::condition_variable* cond)
	{
		int N = 1000000;
		storeElement* elem = nullptr;
		for(int i=0; i<N; i++)
		{
			elem = new storeElement(tag, metric, i, std::sin(i/100.0f*M_PI));
			Task_Pointer tp(new Task(1, Insert, elem, 1, cond));
			controller->ExecuteTask(tp);
		}
	}

	void insertDataCos(StoreController* controller, int metric, int tag, boost::condition_variable* cond)
	{
		int N = 1000000;
		storeElement* elem = nullptr;
		for(int i=0; i<N; i++)
		{
			elem = new storeElement(tag, metric, i, std::cos(i/100.0f*M_PI));
			Task_Pointer tp(new Task(1, Insert, elem, 1, cond));
			controller->ExecuteTask(tp);
		}
	}

	void* createSumSeriesQueryData()
	{
		size_t size = sizeof(int32_t)*11;
		void* query = nullptr; query = malloc(size);
		int tagSize = 2, metricSize = 2, timeSize = 1;
		int tag1 = 1, metric1 = 1;
		int tag2 = 2, metric2 = 2;
		int aggregationType = 13;
		int timePoints = 9999;
		int timePeriodStart = 0;
		int timePeriodEnd = 1000000;
		int position = 0;
		memcpy((char*)query+position, &metricSize, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &metric1, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &metric2, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &tagSize, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &tag1, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &tag2, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &timeSize, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &timePeriodStart, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &timePeriodEnd, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &aggregationType, sizeof(int32_t));
		position+=sizeof(int32_t);
		memcpy((char*)query+position, &timePoints, sizeof(int32_t));
		return query;
	}

	void selectDataSumSeries(
			StoreController* controller,
			int metric1,
			int metric2,
			int tag1,
			int tag2,
			boost::condition_variable* cond,
			int N)
	{
		void* query = createSumSeriesQueryData();
		void* queryCopy = nullptr;
		size_t size = sizeof(int32_t)*11;
		for(int i=0; i<N; i++)
		{
			queryCopy = malloc(size);
			memcpy(queryCopy, query, size);
			Task_Pointer tp(new Task(1, Select, queryCopy, 1, cond));
			controller->ExecuteTask(tp);
		}
		free(query);
	}

	TEST_P(StorePerformance, SelectWithSeriesSum_While_InsertingLinearData_By_Two_Threads)
	{
		int N = GetParam()/10;
		int X = 20;
		duration<double, milli> D;
		for(int i=0; i<X; i++)
		{
			boost::thread insertThreadSin(boost::bind(insertDataSin, this->_storeController, 1, 1, &this->_taskCond));
			boost::thread insertThreadCos(boost::bind(insertDataCos, this->_storeController, 2, 2, &this->_taskCond));
			boost::thread selectThreadSumSeries(boost::bind(selectDataSumSeries,
					this->_storeController, 1, 2, 1, 2, &this->_taskCond, N));
			auto start = system_clock::now();
			insertThreadSin.join();
			insertThreadCos.join();
			selectThreadSumSeries.join();
			delete this->_storeController;
			this->_storeController = nullptr;
			auto end = system_clock::now();
			D += end - start;
			_storeController = new store::StoreController(this->_devId);
		}
		printf("[SelectWithSeriesSumWhileInsertingLinearData] (%f ms)\n", D.count()/X);
		_resultFile << "SelectWithSeriesSumWhileInsertingLinearData " << N << " " << D.count()/X << std::endl;
	}
/*
	TEST_P(StorePerformance, InsertData_EqualElements)
	{
		int N = GetParam();
		int X = 5;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		for(int i=0; i<X; i++)
		{
			storeElement* elem = new storeElement(1, 1, 1, 69.69);
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
		}
		printf("[InsertEqualElem|%d] (%f ms)\n", N, D.count()/X);
		_resultFile << "InsertEqualElem " << N << " " << threadCount << " "
				<< bufferSize << " " << D.count()/X << std::endl;
	}

	TEST_P(StorePerformance, InserData_LinearElements_10000ElemsManyTimes)
	{
		int N = 10000;
		int X = 100;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		for(int i=0; i<X; i++)
		{
			auto start = system_clock::now();
			for(int j=0; j<N; j++)
			{
				storeElement* elem = new storeElement(i%3, i%2, i, i*69.69);
				Task_Pointer tp(new Task(1, Insert, elem, N, &_taskCond));
				_storeController->ExecuteTask(tp);
			}
			delete this->_storeController;
			auto end = system_clock::now();
			D += end - start;
			_storeController = new store::StoreController(this->_devId);
		}
		printf("[InserDataLinearElements|%d] (%f ms)\n", N, D.count()/X);
		_resultFile << "InsertEqualElem " << N << " " << threadCount << " "
				<< bufferSize << " " << D.count()/X << std::endl;
	}

	TEST_P(StorePerformance, CompressionRatioTest_Simple)
	{
		int N = GetParam();
		int X = 2;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		size_t size = 0;
		for(int i=0; i<X; i++)
		{
			auto start = system_clock::now();
			for(int j=0; j<N; j++)
			{
				storeElement* elem = new storeElement(i%3, i%2, i, i*69.69);
				Task_Pointer tp(new Task(1, Insert, elem, N, &_taskCond));
				_storeController->ExecuteTask(tp);
			}
			Task_Pointer flushTp(new Task(2, Flush, nullptr, 1, &_taskCond));
			this->_storeController->ExecuteTask(flushTp);
			size += this->_storeController->GetUsedMemory();
			delete this->_storeController;
			auto end = system_clock::now();
			D += end - start;
			_storeController = new store::StoreController(this->_devId);
		}
		size /= X;
		printf("[CompressionRatioTest_Simple|%d] time: %f ms, compression: from %d B to %lu B => ratio: %f\n", N, D.count()/X, N*24, size, ((float)N*24)/(float)size);
		_resultFile << "CompressionRatioTest_Simple " << N
				<< " " << threadCount
				<< " " << bufferSize
				<< " " << D.count()/X
				<< " " << N*24
				<< " " << size
				<< " " << ((float)N*24)/(float)size
				<< std::endl;
	}

	TEST_P(StorePerformance, CompressionRatioTest_Pessimistic_DefaultUniformRandom_1000Tags_10Metrics)
	{
		int N = GetParam();
		int X = 2;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		size_t size = 0;
		std::default_random_engine generator;
		std::minstd_rand generator2;
		std::minstd_rand generator3;
		std::uniform_int_distribution<int> distribution(0, std::numeric_limits<int>::max());
		std::uniform_int_distribution<int> indexTags(0, 999);
		std::uniform_int_distribution<int> indexMetrics(0, 9);
		for(int i=0; i<X; i++)
		{
			int tags[1000], metrics[10];
			for(int k=0; k<1000; k++)
				tags[k] = distribution(generator);
			for(int l=0; l<10; l++)
				metrics[l] = distribution(generator);

			auto start = system_clock::now();
			for(int j=0; j<N; j++)
			{
				int tag = tags[indexTags(generator2)];
				int metric = metrics[indexMetrics(generator3)];
				storeElement* elem = new storeElement(tag, metric, distribution(generator), i*69.69);
				Task_Pointer tp(new Task(1, Insert, elem, 1, &_taskCond));
				_storeController->ExecuteTask(tp);
			}
			Task_Pointer flushTp(new Task(2, Flush, nullptr, 1, &_taskCond));
			this->_storeController->ExecuteTask(flushTp);
			size += this->_storeController->GetUsedMemory();
			delete this->_storeController;
			auto end = system_clock::now();
			D += end - start;

			_storeController = new store::StoreController(this->_devId);
		}
		size /= X;
		printf("[CompressionRatioTest_Pessimistic_NElems|%d] time: %f ms, compression: from %d B to %lu B => ratio: %f\n", N, D.count()/X, N*24, size, ((float)N*24)/(float)size);
		_resultFile << "CompressionRatioTest_Pessimistic_NElems " << N
				<< " " << threadCount
				<< " " << bufferSize
				<< " " << D.count()/X
				<< " " << N*24
				<< " " << size
				<< " " << ((float)N*24)/(float)size
				<< std::endl;
	}

	TEST_P(StorePerformance, CompressionRatioTest_Pessimistic_DefaultUniformRandom_NTags_20Metrics_1MElems)
	{
		int N = 1000000;
		int T = GetParam();
		int X = 2;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		size_t size = 0;
		std::default_random_engine generator;
		std::minstd_rand generator2;
		std::minstd_rand generator3;
		std::uniform_int_distribution<int> distribution(0, std::numeric_limits<int>::max());
		std::uniform_int_distribution<int> indexTags(0, T-1);
		std::uniform_int_distribution<int> indexMetrics(0, 19);
		int* tags = new int[T];
		for(int i=0; i<X; i++)
		{
			int metrics[20];
			for(int k=0; k<T; k++)
				tags[k] = distribution(generator);
			for(int l=0; l<20; l++)
				metrics[l] = distribution(generator);

			auto start = system_clock::now();
			for(int j=0; j<N; j++)
			{
				int tag = tags[indexTags(generator2)];
				int metric = metrics[indexMetrics(generator3)];
				storeElement* elem = new storeElement(tag, metric, distribution(generator), i*69.69);
				Task_Pointer tp(new Task(1, Insert, elem, 1, &_taskCond));
				_storeController->ExecuteTask(tp);
			}
			Task_Pointer flushTp(new Task(2, Flush, nullptr, 1, &_taskCond));
			this->_storeController->ExecuteTask(flushTp);
			size += this->_storeController->GetUsedMemory();
			delete this->_storeController;
			auto end = system_clock::now();
			D += end - start;

			_storeController = new store::StoreController(this->_devId);
		}
		delete [] tags;
		size /= X;
		printf("[CompressionRatioTest_Pessimistic_NTrunks|%d] time: %f ms, compression: from %d B to %lu B => ratio: %f\n", T, D.count()/X, N*24, size, ((float)N*24)/(float)size);
		_resultFile << "CompressionRatioTest_Pessimistic_NTrunks " << N
				<< " " << threadCount
				<< " " << bufferSize
				<< " " << D.count()/X
				<< " " << N*24
				<< " " << size
				<< " " << ((float)N*24)/(float)size
				<< std::endl;
	}

	TEST_P(StorePerformance, CompressionRatioTest_SuperPessimistic_DefaultUniformRandom)
	{
		if(GetParam()>10000) return;
		int N = GetParam();
		int X = 2;
		int threadCount = _config->GetIntValue("INSERT_THREAD_POOL_SIZE");
		int bufferSize = _config->GetIntValue("STORE_BUFFER_CAPACITY");
		duration<double, milli> D;
		size_t size = 0;
		std::default_random_engine generator;
		std::minstd_rand generator2;
		std::minstd_rand generator3;
		std::uniform_int_distribution<int> tags(0, std::numeric_limits<int>::max());
		std::uniform_int_distribution<int> metrics(0, std::numeric_limits<int>::max());
		std::uniform_int_distribution<int32_t> distribution64(0, std::numeric_limits<int32_t>::max());
		for(int i=0; i<X; i++)
		{
			auto start = system_clock::now();
			for(int j=0; j<N; j++)
			{
				int tag = tags(generator2);
				int metric = metrics(generator3);
				storeElement* elem = new storeElement(tag, metric, distribution64(generator), i*69.69);
				Task_Pointer tp(new Task(1, Insert, elem, 1, &_taskCond));
				_storeController->ExecuteTask(tp);
			}
			Task_Pointer flushTp(new Task(2, Flush, nullptr, 1, &_taskCond));
			this->_storeController->ExecuteTask(flushTp);
			size += this->_storeController->GetUsedMemory();
			delete this->_storeController;
			auto end = system_clock::now();
			D += end - start;
			_storeController = new store::StoreController(this->_devId);
		}
		size /= X;
		printf("[CompressionRatioTest_SuperPessimistic|%d] time: %f ms, compression: from %d B to %lu B => ratio: %f\n", N, D.count()/X, N*24, size, ((float)N*24)/(float)size);
		_resultFile << "CompressionRatioTest_SuperPessimistic " << N
				<< " " << threadCount
				<< " " << bufferSize
				<< " " << D.count()/X
				<< " " << N*24
				<< " " << size
				<< " " << ((float)N*24)/(float)size
				<< std::endl;
	}*/

} /* namespace task */
} /* namespace ddj */
