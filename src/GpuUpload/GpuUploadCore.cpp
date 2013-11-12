/*
 * GpuUploaderCore.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "GpuUploadCore.h"


namespace ddj {
namespace store {

	void GpuUploadCore::CopyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, cudaStream_t stream)
	{
		CUDA_CHECK_RETURN
		(
				cudaMemcpyAsync
				(
						(void*)devicePointer,
						(void*)hostPointer,
						(size_t) numElements * sizeof(storeElement),
						cudaMemcpyHostToDevice,
						stream
				)
		);
	}

	void GpuUploadCore::AppendToMainStore(void* devicePointer, size_t size, infoElement* info)
	{
		cudaStream_t stream = this->_cudaController->GetSyncStream();
		info->startValue = this->_cudaController->GetMainMemoryOffset();
		CUDA_CHECK_RETURN
				(
						cudaMemcpyAsync
						(
								this->_cudaController->GetMainMemoryFirstFreeAddress(),
								devicePointer,
								size,
								cudaMemcpyDeviceToDevice,
								stream
						)
				);
		info->endValue = info->startValue + size;
		this->_cudaController->SetMainMemoryOffset(info->endValue);
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream) );
	}

	size_t GpuUploadCore::CompressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream)
	{
		size_t size = sizeof(storeElement)*elemToUploadCount;
		CUDA_CHECK_RETURN( cudaMalloc(result, size) );
		CUDA_CHECK_RETURN
						(
								cudaMemcpyAsync
								(
										*result,
										deviceBufferPointer,
										size,
										cudaMemcpyDeviceToDevice,
										stream
								)
						);
		return size;
	}

	GpuUploadCore::GpuUploadCore(CudaController* cudaController)
	{
		this->_cudaController = cudaController;
	}

	GpuUploadCore::~GpuUploadCore(){}

} /* namespace store */
} /* namespace ddj */
