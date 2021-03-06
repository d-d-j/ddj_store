/*
 * CudaController.h
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#ifndef CUDACONTROLLER_H_
#define CUDACONTROLLER_H_

#include "CudaIncludes.h"
#include "CudaCommons.h"
#include "../Core/Semaphore.h"
#include "../Core/Config.h"
#include "../Core/Logger.h"
#include "../Store/StoreTypedefs.h"
#include <boost/thread.hpp>
#include <boost/lockfree/queue.hpp>

namespace ddj {
namespace store {

	class CudaController : public boost::noncopyable
	{
		int _cudaDeviceId;

		/* STREAMS */
		Semaphore* _uploadStreamsSemaphore;
		Semaphore* _queryStreamsSemaphore;
		boost::lockfree::queue<cudaStream_t>* _uploadStreams;
		boost::lockfree::queue<cudaStream_t>* _queryStreams;
		cudaStream_t _syncStream;

		/* MAIN STORE MEMORY (on GPU) */
		ullint _mainMemoryOffset;
		boost::mutex _offsetMutex;
		void* _mainMemoryPointer;
		ullint _mainMemoryCapacity;

		/* LOGGER & CONFIG & CUDA_COMMONS */
		Logger _logger = Logger::getRoot();
		Config* _config = Config::GetInstance();
		CudaCommons _cudaCommons;

	public:
		CudaController(int uploadStreamsNum, int queryStreamsNum, int cudaDeviceId);
		virtual ~CudaController();
		int GetCudaDeviceId();
		void SetProperDevice();

		/* STREAMS */
		cudaStream_t GetUploadStream();
		cudaStream_t GetQueryStream();
		void ReleaseUploadStream(cudaStream_t st);
		void ReleaseQueryStream(cudaStream_t st);
		cudaStream_t GetSyncStream();

		/* MAIN STORE ARRAY */
		ullint GetMainMemoryOffset();
		void SetMainMemoryOffset(ullint offset);
		void* GetMainMemoryPointer();
		void* GetMainMemoryFirstFreeAddress();
		ullint GetMainMemoryCapacity();

	private:
		void allocateMainGpuStorage();
	};

} /* namespace store */
} /* namespace ddj */
#endif /* CUDACONTROLLER_H_ */
