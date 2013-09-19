/*
 * GpuUploaderMonitor.h
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#ifndef GPUUPLOADERMONITOR_H_
#define GPUUPLOADERMONITOR_H_

#include "GpuUploaderCore.h"
#include "../BTree/BTreeMonitor.h"

namespace ddj {
namespace store {

	class GpuUploaderMonitor
	{
		private:
			boost::thread* _threadUploader;

			boost::condition_variable _condUploader;
			boost::condition_variable _condSynchronization;

			boost::mutex _mutexSynchronization;
			boost::mutex _mutexUploader;

			BTreeMonitor* _bTreeInserter;

			volatile sig_atomic_t _readyToUpload;	/**< Whether or not elements were copied to queue */

			//TODO: Change this to ringbuffer from boost
			storeElement* _elementsToUpload;

		public:
			GpuUploaderMonitor(BTreeMonitor* bTreeInserter);
			virtual ~GpuUploaderMonitor();
			bool SendStoreElementsToGpu(boost::array<storeElement, STORE_BUFFER_SIZE>* elements);
		private:
			void threadFunction();	//!< a function which is being executed by _threadUploader thread
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADERMONITOR_H_ */
