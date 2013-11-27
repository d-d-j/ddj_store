/*
 * GpuUploaderCore.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "GpuUploadCore.h"


namespace ddj {
namespace store {

	storeTrunkInfo* GpuUploadMonitor::Upload
			(
			boost::array<storeElement, STORE_BUFFER_SIZE>* elements,
			int elementsToUploadCount
			)
	{
		cudaStream_t stream = this->_cudaController->GetUploadStream();

		storeTrunkInfo* result = new storeTrunkInfo(elements->front().metric, elements->front().time, elements->back().time, 0, 0);

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

	void StoreUploadCore::CopyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, cudaStream_t stream)
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

	void StoreUploadCore::AppendToMainStore(void* devicePointer, size_t size, storeTrunkInfo* info)
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

	size_t StoreUploadCore::CompressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream)
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

	StoreUploadCore::StoreUploadCore(CudaController* cudaController)
	{
		this->_cudaController = cudaController;
	}

	StoreUploadCore::~StoreUploadCore(){}

} /* namespace store */
} /* namespace ddj */
