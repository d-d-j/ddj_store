/*
 * GpuUploaderCore.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "GpuUploadCore.h"


namespace ddj {
namespace store {

	void GpuUploadCore::CopyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, int streamNum)
	{
		cudaStream_t st = this->_cudaController->GetUploadStream(streamNum);
		CUDA_CHECK_RETURN
		(
				cudaMemcpyAsync
				(
						(void*)devicePointer,
						(void*)hostPointer,
						(size_t) numElements * sizeof(storeElement),
						cudaMemcpyHostToDevice,
						st
				)
		);
	}

	void GpuUploadCore::AppendToMainStore(void* devicePointer, size_t size, infoElement* info)
	{
		info->startValue = this->_cudaController->GetMainMemoryOffset();
		CUDA_CHECK_RETURN
				(
						cudaMemcpyAsync
						(
								this->_cudaController->GetMainMemoryFirstFreeAddress(),
								devicePointer,
								size,
								cudaMemcpyDeviceToDevice,
								this->_cudaController->GetUploadStream(0)
						)
				);
		info->endValue = info->startValue + size;
		this->_cudaController->SetMainMemoryOffset(info->endValue);
	}

	size_t GpuUploadCore::CompressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, int streamNum, void** result)
	{
		*result = deviceBufferPointer;
		return elemToUploadCount * sizeof(storeElement);
	}

	GpuUploadCore::GpuUploadCore(CudaController* cudaController)
	{
		this->_cudaController = cudaController;
	}

	GpuUploadCore::~GpuUploadCore(){}

} /* namespace store */
} /* namespace ddj */
