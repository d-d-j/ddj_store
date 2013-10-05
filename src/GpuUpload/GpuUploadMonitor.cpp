/*
 * GpuUploadMonitor.cpp
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#include "GpuUploadMonitor.h"

namespace ddj {
namespace store {


GpuUploadMonitor::GpuUploadMonitor() {

	this ->_core = GpuUploadCore(2);
	CUDA_CHECK_RETURN(cudaMalloc((void**) &this->devicePointer, STORE_BUFFER_SIZE * sizeof(storeElement)));
	h_LogThreadDebug("Monitor device malloc");
}


infoElement* GpuUploadMonitor::Upload(
		boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
		int elementsToUploadCount) {

	h_LogThreadDebug("Monitor Upload starting");

	boost::mutex::scoped_lock lock(this->_mutex);

	_core.copyToGpu((storeElement*)elements, (storeElement*)this->devicePointer, elementsToUploadCount);

	return new infoElement(1, 1, 1, 1, 1);
}

} /* namespace store */
} /* namespace ddj */
