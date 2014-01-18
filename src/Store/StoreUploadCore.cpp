/*
 * GpuUploaderCore.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "StoreUploadCore.h"
#include <algorithm>

namespace ddj {
namespace store {

	StoreUploadCore::StoreUploadCore(CudaController* cudaController)
		: _logger(Logger::getRoot()),_config(Config::GetInstance())
	{
		this->_cudaController = cudaController;
		this->_enableCompression = this->_config->GetIntValue("ENABLE_COMPRESSION");
	}

	StoreUploadCore::~StoreUploadCore(){}

	storeTrunkInfo* StoreUploadCore::sortTrunkAndPrepareInfo(storeElement* elementsToUpload, int elementsToUploadCount)
	{
		std::sort(elementsToUpload, elementsToUpload+elementsToUploadCount, OrderStoreElementByTimeAsc());
		return new storeTrunkInfo(
						elementsToUpload[0].metric,
						elementsToUpload[0].time,
						elementsToUpload[elementsToUploadCount-1].time,
						0,
						0);
	}

	storeTrunkInfo* StoreUploadCore::Upload(storeElement* elementsToUpload, int elementsToUploadCount)
	{
		if(elementsToUploadCount == 0) return nullptr;

		// GET CUDA STREAM
		cudaStream_t stream = this->_cudaController->GetUploadStream();

		// SORT TRUNK AND SET STORE TRUNK INFO
		storeTrunkInfo* result = this->sortTrunkAndPrepareInfo(elementsToUpload, elementsToUploadCount);

		// ALLOC DEVICE BUFFER
		storeElement* deviceBufferPointer = nullptr;
		CUDA_CHECK_RETURN( cudaMalloc((void**) &(deviceBufferPointer), elementsToUploadCount*sizeof(storeElement)) );

		// COPY BUFFER TO GPU
		copyToGpu(elementsToUpload, deviceBufferPointer, elementsToUploadCount, stream);

		// COMPRESSION (returns pointer to new memory)
		void* compressedBufferPointer = deviceBufferPointer;
		size_t size = sizeof(storeElement)*elementsToUploadCount;
		if(this->_enableCompression)
			size = compressGpuBuffer(deviceBufferPointer, elementsToUploadCount, &compressedBufferPointer, stream);

		// AFTER GPU BUFFER COMPRESSION WE CAN REUSE STREAM AND RELEASE DEVICE BUFFER
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream) );
		this->_cudaController->ReleaseUploadStream(stream);
		if(this->_enableCompression) CUDA_CHECK_RETURN( cudaFree(deviceBufferPointer) );

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
		int newOffset = info->startValue + size;
		info->endValue = newOffset - 1;
		this->_cudaController->SetMainMemoryOffset(newOffset);
		CUDA_CHECK_RETURN( cudaStreamSynchronize(stream) );
	}

	size_t StoreUploadCore::compressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream)
	{
		size_t size = sizeof(storeElement)*elemToUploadCount;
		compression::Compression c;
		return c.CompressTrunk(deviceBufferPointer, size, result);
	}

} /* namespace store */
} /* namespace ddj */
