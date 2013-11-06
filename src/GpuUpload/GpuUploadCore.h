/*
 * GpuUploaderCore.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef GPUUPLOADCORE_H_
#define GPUUPLOADCORE_H_

#include "../Store/storeElement.h"
#include "../Store/infoElement.h"
#include "cuda_runtime.h"
#include "../CUDA/cudaIncludes.h"
#include "../CUDA/CudaController.h"

namespace ddj
{
namespace store
{

	class GpuUploadCore
	{
		CudaController* _cudaController;
	public:
		GpuUploadCore(CudaController* cudaController);
		virtual ~GpuUploadCore();

		void CopyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, int streamNum);
		size_t CompressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, int streamNum, void** result);
		void AppendToMainStore(void* devicePointer, size_t size, infoElement* info);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADCORE_H_ */
