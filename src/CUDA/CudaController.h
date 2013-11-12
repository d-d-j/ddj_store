/*
 * CudaController.h
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#ifndef CUDACONTROLLER_H_
#define CUDACONTROLLER_H_

#include "GpuStore.cuh"
#include "../Store/StoreIncludes.h"
#include "../Helpers/Logger.h"
#include <boost/lockfree/queue.hpp>
#include "cudaIncludes.h"
#include "../Helpers/Semaphore.h"

namespace ddj {
namespace store {

	class CudaController : public boost::noncopyable
	{
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

		/* LOGGER */
		Logger _logger = Logger::getRoot();
	public:
		CudaController(int uploadStreamsNum, int queryStreamsNum);
		virtual ~CudaController();

		/* STREAMS */
		cudaStream_t GetUploadStream();
		cudaStream_t GetQueryStream();
		void ReleaseUploadStream(cudaStream_t st);
		void ReleaseQueryStream(cudaStream_t st);
		cudaStream_t GetSyncStream();

		/* MAIN STORE MEMORY (on GPU) */
		ullint GetMainMemoryOffset();
		void SetMainMemoryOffset(ullint offset);
		void* GetMainMemoryPointer();
		void* GetMainMemoryFirstFreeAddress();
	};

} /* namespace store */
} /* namespace ddj */
#endif /* CUDACONTROLLER_H_ */
