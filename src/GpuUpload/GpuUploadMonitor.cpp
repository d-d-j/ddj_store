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
		this->_cudaController = cudaController;

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor constructor [END]"));
	}

	GpuUploadMonitor::~GpuUploadMonitor()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor destructor [BEGIN]"));

		delete this->_core;

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Gpu upload monitor destructor [END]"));
	}

	infoElement* GpuUploadMonitor::Upload
			(
			boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
			int elementsToUploadCount
			)
	{
		cudaStream_t stream = this->_cudaController->GetUploadStream();

		infoElement* result = new infoElement(elements->front().tag, elements->front().time, elements->back().time, 0, 0);

		storeElement* deviceBufferPointer;

		CUDA_CHECK_RETURN( cudaMalloc((void**) &(deviceBufferPointer), STORE_BUFFER_SIZE*sizeof(storeElement)) );

		storeElement* elementsToUpload = elements->c_array();

		// COPY BUFFER TO GPU
		_core->CopyToGpu(elementsToUpload, deviceBufferPointer, elementsToUploadCount, stream);

		// TODO: NOW BUFFER CAN BE SWAPPED AGAIN...

		// COMPRESSION (returns pointer to new memory)
		void* compressedBufferPointer;
		size_t size = _core->CompressGpuBuffer(deviceBufferPointer, elementsToUploadCount, &compressedBufferPointer, stream);

		// AFTER GPU BUFFER COMPRESSION WE CAN REUSE STREAM AND RELEASE DEVICE BUFFER
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream) );
		this->_cudaController->ReleaseUploadStream(stream);
		CUDA_CHECK_RETURN( cudaFree(deviceBufferPointer) );

		// APPEND UPLOADED BUFFER TO MAIN GPU STORE (IN STREAM 0)
		{
			boost::mutex::scoped_lock lock(this->_mutex);
			_core->AppendToMainStore(compressedBufferPointer, size, result);
		}

		// RELEASE COMPRESSED DEVICE BUFFER
		CUDA_CHECK_RETURN( cudaFree(compressedBufferPointer) );

		// RETURN INFORMATION ABOUT UPLOADED BUFFER LOCATION IN MAIN GPU STORE
		return result;
	}

} /* namespace store */
} /* namespace ddj */
