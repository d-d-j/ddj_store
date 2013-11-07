/*
 * GpuUploadMonitor.cpp
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#include "GpuUploadMonitor.h"

namespace ddj {
namespace store {

	GpuUploadMonitor::GpuUploadMonitor(CudaController* cudaController)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor constructor [BEGIN]"));

		this->_core = new GpuUploadCore(cudaController);
		this->_sem = new Semaphore(DEVICE_BUFFERS_COUNT);

		// ALLOCATE GPU STORE BUFFERS
		for(int i=0; i<DEVICE_BUFFERS_COUNT; i++)
		{
			CUDA_CHECK_RETURN
					(
					cudaMalloc((void**) &(this->_deviceBufferPointers[i]), STORE_BUFFER_SIZE*sizeof(storeElement))
					);
		}

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor constructor [END]"));
	}

	GpuUploadMonitor::~GpuUploadMonitor()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor destructor [BEGIN]"));

		delete this->_sem;
		delete this->_core;

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor destructor [END]"));
	}

	infoElement* GpuUploadMonitor::Upload
			(
			boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
			int elementsToUploadCount
			)
	{
		// TODO: HANDLE ERRORS HERE

		int streamNum = this->_sem->Wait();

		infoElement* result = new infoElement(elements->front().tag, elements->front().time, elements->back().time, 0, 0);

		storeElement* deviceBufferPointer = this->_deviceBufferPointers[streamNum-1];

		storeElement* elementsToUpload = elements->c_array();

		// COPY BUFFER TO GPU
		_core->CopyToGpu(elementsToUpload, deviceBufferPointer, elementsToUploadCount, streamNum);

		// TODO: NOW BUFFER CAN BE SWAPPED AGAIN...
		void* compressedBufferPointer;
		size_t size = _core->CompressGpuBuffer(deviceBufferPointer, elementsToUploadCount, streamNum, &compressedBufferPointer);

		// AFTER GPU BUFFER COMPRESSION WE CAN REUSE STREAM
		this->_sem->Release();

		// APPEND UPLOADED BUFFER TO MAIN GPU STORE (IN STREAM 0)
		_core->AppendToMainStore(compressedBufferPointer, size, result);

		// RETURN INFORMATION ABOUT UPLOADED BUFFER LOCATION IN MAIN GPU STORE
		return result;
	}

} /* namespace store */
} /* namespace ddj */
