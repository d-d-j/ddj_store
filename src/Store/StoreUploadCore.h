/*
 * GpuUploaderCore.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef GPUUPLOADCORE_H_
#define GPUUPLOADCORE_H_

#include "StoreTrunkInfo.h"
#include "StoreElement.cuh"
#include "../Cuda/CudaController.h"
#include "../Compression/Compression.h"
#include <boost/thread.hpp>

namespace ddj {
namespace store {

	class StoreUploadCore
	{
		CudaController* _cudaController;
		boost::mutex _mutex;
		bool _enableCompression;

		/* LOGGER & CONFIG */
		Logger _logger;
		Config* _config;

	public:
		StoreUploadCore(CudaController* cudaController);
		virtual ~StoreUploadCore();
		storeTrunkInfo* Upload(storeElement* elements, int elementsToUploadCount);

	private:
		void copyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, cudaStream_t stream);
		size_t compressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream);
		void appendToMainStore(void* devicePointer, size_t size, storeTrunkInfo* info);
		storeTrunkInfo* sortTrunkAndPrepareInfo(storeElement* elementsToUpload, int elementsToUploadCount);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADCORE_H_ */
