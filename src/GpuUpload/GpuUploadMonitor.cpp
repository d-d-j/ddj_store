/*
 * GpuUploadMonitor.cpp
 *
 *  Created on: Aug 31, 2013
 *      Author: Karol Dzitkowski
 */

#include "GpuUploadMonitor.h"

namespace ddj
{
namespace store
{

GpuUploadMonitor::GpuUploadMonitor(CudaController* cudaController)
{
	Config* config = Config::GetInstance();
	h_LogThreadDebug("Gpu upload monitor constructor started");
	this->_core = new GpuUploadCore(cudaController);
	this->_sem = new Semaphore(config->GetIntValue("DEVICE_BUFFERS_COUNT"));

	// ALLOCATE GPU STORE BUFFERS
	h_LogThreadDebug("Gpu upload monitor allocates store buffer on GPU");
	for (int i = 0; i < DEVICE_BUFFERS_COUNT; i++)
	{
		CUDA_CHECK_RETURN(
				cudaMalloc((void** ) &(this->_deviceBufferPointers[i]),
						STORE_BUFFER_SIZE
								* sizeof(storeElement)));
	}

	h_LogThreadDebug("Gpu upload monitor constructor ended");
}

GpuUploadMonitor::~GpuUploadMonitor()
{
	delete this->_sem;
	delete this->_core;
}

infoElement* GpuUploadMonitor::Upload(
		boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
		int elementsToUploadCount)
{
	h_LogThreadDebug("GpuUploadMonitor::Upload [BEGIN]");
	// TODO: HANDLE ERRORS HERE... How we will do that?

	int streamNum = this->_sem->Wait();

	infoElement* result = new infoElement(elements->front().tag,
			elements->front().time, elements->back().time, 0, 0);

	storeElement* deviceBufferPointer = this->_deviceBufferPointers[streamNum
			- 1];

	storeElement* elementsToUpload = elements->c_array();

	// COPY BUFFER TO GPU
	_core->CopyToGpu(elementsToUpload, deviceBufferPointer,
			elementsToUploadCount, streamNum);

	// TODO: NOW BUFFER CAN BE SWAPPED AGAIN...
	void* compressedBufferPointer;
	size_t size = _core->CompressGpuBuffer(deviceBufferPointer,
			elementsToUploadCount, streamNum, &compressedBufferPointer);

	// AFTER GPU BUFFER COMPRESSION WE CAN REUSE STREAM
	this->_sem->Release();

	// APPEND UPLOADED BUFFER TO MAIN GPU STORE (IN STREAM 0)
	_core->AppendToMainStore(compressedBufferPointer, size, result);

	h_LogThreadDebug("GpuUploadMonitor::Upload [END]");

	// RETURN INFORMATION ABOUT UPLOADED BUFFER LOCATION IN MAIN GPU STORE
	return result;
}

} /* namespace store */
} /* namespace ddj */
