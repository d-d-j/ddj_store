/*
 * CudaController.h
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#ifndef CUDACONTROLLER_H_
#define CUDACONTROLLER_H_

#include "../Store/StoreIncludes.h"
#include "../Helpers/Config.h"
#include "../Helpers/Logger.h"
#include <boost/lockfree/queue.hpp>
#include "cudaIncludes.h"
#include "../Helpers/Semaphore.h"
#include "../CUDA/CudaCommons.h"

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

		/* LOGGER & CONFIG & CUDA_COMMONS */
		Logger _logger = Logger::getRoot();
		Config* _config = Config::GetInstance();
		CudaCommons _cudaCommons;

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
		cudaStream_t GetUploadStream(int num);
		cudaStream_t GetQueryStream(int num);

		/* MAIN STORE ARRAY */
		ullint GetMainMemoryOffset();
		void SetMainMemoryOffset(ullint offset);
		void* GetMainMemoryPointer();
		void* GetMainMemoryFirstFreeAddress();

	private:
		void allocateMainGpuStorage();
	};

} /* namespace store */
} /* namespace ddj */
#endif /* CUDACONTROLLER_H_ */
