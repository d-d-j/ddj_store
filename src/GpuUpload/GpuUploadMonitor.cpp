/*
 * GpuUploadMonitor.cpp
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#include "GpuUploadMonitor.h"

namespace ddj {
namespace store {

	infoElement* GpuUploadMonitor::Upload
									(
									boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
									int elementsToUploadCount
									)
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		// TODO: IMPLEMENT GpuUploadCore.Upload(...) - ergo uploading elements to GPU

		// Here is the place for:

		// some result = _core.Upload(...)

		// return result as infoElement*;

		return NULL;
	}

} /* namespace store */
} /* namespace ddj */
