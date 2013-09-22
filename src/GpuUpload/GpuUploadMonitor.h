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

namespace ddj {
namespace store {

	class GpuUploadMonitor
	{
		private:
			boost::mutex _mutex;
			GpuUploadCore _core;

		public:
			infoElement* Upload
							(
							boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
							int elementsToUploadCount
							);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADMONITOR_H_ */
