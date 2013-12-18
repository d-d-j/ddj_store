/*
 * GpuUploaderCore.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "StoreUploadCore.h"


namespace ddj {
namespace store {

	StoreUploadCore::StoreUploadCore(CudaController* cudaController)
	{
		this->_cudaController = cudaController;
	}

	StoreUploadCore::~StoreUploadCore(){}



	storeTrunkInfo* StoreUploadCore::Upload(storeElement* elementsToUpload, int elementsToUploadCount)
	{
		// GET CUDA STREAM
		cudaStream_t stream = this->_cudaController->GetUploadStream();

		storeTrunkInfo* result = new storeTrunkInfo(
				elementsToUpload[0].metric,
				elementsToUpload[0].time,
				elementsToUpload[elementsToUploadCount-1].time,
				0,
				0);

		// ALLOC DEVICE BUFFER
		storeElement* deviceBufferPointer = nullptr;
		CUDA_CHECK_RETURN( cudaMalloc((void**) &(deviceBufferPointer), elementsToUploadCount*sizeof(storeElement)) );

		// COPY BUFFER TO GPU
		copyToGpu(elementsToUpload, deviceBufferPointer, elementsToUploadCount, stream);

		// TODO: SORT ARRAY ON GPU AND RETURN PROPER storeTrunkInfo (because this one above has wrong start/end time)

		// COMPRESSION (returns pointer to new memory)
		void* compressedBufferPointer;
		size_t size = compressGpuBuffer(deviceBufferPointer, elementsToUploadCount, &compressedBufferPointer, stream);

		// AFTER GPU BUFFER COMPRESSION WE CAN REUSE STREAM AND RELEASE DEVICE BUFFER
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream) );
		this->_cudaController->ReleaseUploadStream(stream);
		CUDA_CHECK_RETURN( cudaFree(deviceBufferPointer) );

		// APPEND UPLOADED BUFFER TO MAIN GPU STORE (IN STREAM 0)
		{
			boost::mutex::scoped_lock lock(this->_mutex);
			appendToMainStore(compressedBufferPointer, size, result);
		}

		// RELEASE COMPRESSED DEVICE BUFFER
		CUDA_CHECK_RETURN( cudaFree(compressedBufferPointer) );

		// RETURN INFORMATION ABOUT UPLOADED BUFFER LOCATION IN MAIN GPU STORE
		return result;
	}

	void StoreUploadCore::copyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, cudaStream_t stream)
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

	void StoreUploadCore::appendToMainStore(void* devicePointer, size_t size, storeTrunkInfo* info)
	{
		cudaStream_t stream = this->_cudaController->GetSyncStream();
		info->startValue = this->_cudaController->GetMainMemoryOffset();	// index of first byte of trunk in main gpu memory
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
		info->endValue = info->startValue + size - 1;	// index of last byte of trunk in main gpu memory
		this->_cudaController->SetMainMemoryOffset(info->endValue);
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream) );
	}

	size_t StoreUploadCore::compressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream)
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

} /* namespace store */
} /* namespace ddj */
