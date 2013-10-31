/*
 * GpuUploadMonitor.h
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#ifndef GPUUPLOADMONITOR_H_
#define GPUUPLOADMONITOR_H_

#include "GpuUploadCore.h"
#include "../Store/storeElement.h"
#include "../Store/infoElement.h"
#include "../BTree/BTreeMonitor.h"
#include "../Store/storeSettings.h"
#include "../Helpers/Semaphore.h"
#include "../CUDA/CudaController.h"
#include "../Helpers/Config.h"

namespace ddj {
namespace store {

	class GpuUploadMonitor
	{
		private:
			GpuUploadCore* _core;
			Semaphore* _sem;
			boost::array<storeElement*, STREAMS_NUM_UPLOAD> _deviceBufferPointers;

		public:
			GpuUploadMonitor(CudaController* cudaController);
			~GpuUploadMonitor();
			infoElement* Upload
							(
							boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
							int elementsToUploadCount
							);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADMONITOR_H_ */
